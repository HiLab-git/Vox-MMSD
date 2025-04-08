import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F



class DINOLoss_modmask(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ncrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ncrops = ncrops

        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, patch_out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))

    def forward(self, student_output, teacher_output, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out_cls, student_out_modmask = student_output
        teacher_out_cls, teacher_out_modmask = teacher_output


        student_out_cls = student_out_cls / self.student_temp
        student_out_modmask = student_out_modmask / self.student_temp
        student_out_cls_c = student_out_cls.chunk(self.ncrops)
        student_out_modmask_c = student_out_modmask.chunk(self.ncrops)

        student_mask = student_mask.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]

        teacher_out_cls_c = F.softmax((teacher_out_cls - self.center) / temp, dim=-1)
        teacher_out_cls_c = teacher_out_cls_c.detach().chunk(2)
        teacher_out_modmask_c = F.softmax((teacher_out_modmask - self.center2) / temp2, dim=-1)
        teacher_out_modmask_c = teacher_out_modmask_c.detach().chunk(2)

        total_loss_1 = 0
        n_loss_terms_1 = 0
        total_loss_2 = 0
        n_loss_terms_2 = 0
        for iq, q in enumerate(teacher_out_cls_c):
            for iv, v in enumerate(student_out_cls_c):
                if iv == iq:
                    #遮罩重建的loss
                    loss_2 = torch.sum(-teacher_out_modmask_c[iq] * F.log_softmax(student_out_modmask_c[iv], dim=-1), dim=-1)
                    mask = student_mask[iv]
                    loss_2 = torch.sum(loss_2 * mask.float(), dim=(-1,-2,-3)) / mask.sum(dim=(-1,-2,-3)).clamp(min=1.0)
                    total_loss_2 += loss_2.mean()
                    n_loss_terms_2 += 1
                else:
                    #原本DINO的loss
                    loss_1= torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                    total_loss_1 += loss_1.mean()
                    n_loss_terms_1 += 1
        total_loss_1 /= n_loss_terms_1
        total_loss_2 /= n_loss_terms_2
        total_loss = dict(cls=total_loss_1, modmask=total_loss_2, loss=total_loss_1 + total_loss_2)
        self.update_center(teacher_out_cls, teacher_out_modmask)                  
        return total_loss


    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_modmask):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        cls_center = cls_center / len(teacher_cls)
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        modmask_center = torch.sum(teacher_modmask.mean(1), dim=0, keepdim=True)
        modmask_center = modmask_center / len(teacher_modmask)
        self.center2 = self.center2 * self.center_momentum2 + modmask_center * (1 - self.center_momentum2)