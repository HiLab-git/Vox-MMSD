import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from pymic.loss.seg.combined import CombinedLoss
from pymic.loss.loss_dict_seg import SegLossDict
from pymic.loss.seg.gatedcrf import GatedCRFLoss



class DINOLoss_Seg(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, use_CRF=False):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops

        self.use_CRF = use_CRF
        self.kernels_desc = [{'weight': 1,'xy': 5,'rgb': 0.1}]
        self.kernels_radius = 5
        self.crf_loss_calculator = GatedCRFLoss()

        self.register_buffer("center", torch.zeros(48,4,48,48,48))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))


    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0

        #计算CRF LOSS   
        # if(self.use_CRF):
        #     crf_loss = self.crf_loss_calculator(outputs_soft, self.kernels_desc, self.kernels_radius, batch_dict, outputs_soft.shape[-1], outputs_soft.shape[-1])
        #     total_loss += 0.01 * crf_loss['loss']
        
        #计算CE LOSS
        ce_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=1), dim=1)
                ce_loss += loss.mean()
                n_loss_terms += 1
        ce_loss /= n_loss_terms
        total_loss += ce_loss

        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """

        #对于 (B,C,D,H,W)， 在B,D,H,W四个维度上去求均值
        pixel_center = torch.mean(teacher_output, dim=(0, 2, 3, 4), keepdim=True)
    
        # ema update
        self.center = self.center * self.center_momentum + pixel_center * (1 - self.center_momentum)