# -*- coding: utf-8 -*-
from __future__ import print_function, division

import copy
import csv
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import *
from random import random
from torchvision import transforms
from tensorboardX import SummaryWriter
from pymic import TaskType
from pymic.net.net_dict_seg import SegNetDict
from pymic.loss.loss_dict_cls import PyMICClsLossDict
from pymic.net.net_dict_cls import TorchClsNetDict
from pymic.transform.trans_dict import TransformDict
from pymic.transform.view_generator import ContrastiveLearningViewGeneratorVoxMMSD
from pymic.transform.intensity import VoxMMSDMaskedImage
from pymic.transform.normalize import NormalizeWithMinMax
from pymic.io.nifty_dataset import NiftyDataset
from pymic.loss.seg.dinoloss import DINOLoss
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.util.general import mixup, tensor_shape_match
from torch.cuda.amp import GradScaler, autocast
from torch.cuda import amp
from pymic.net_run.self_sup.util import *
import torch.nn.functional as F
import warnings
import torch
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

def select_from_pyramid(
            feature_pyramid: Sequence[torch.Tensor],
            indices: torch.Tensor,
    ) -> torch.Tensor:
        """Select features from feature pyramid by their indices w.r.t. base feature map.

        Args:
            feature_pyramid (Sequence[torch.Tensor]): Sequence of tensors of shapes ``(c_i, h_i, w_i, d_i)``.
            indices (torch.Tensor): tensor of shape ``(n, 3)``

        Returns:
            torch.Tensor: tensor of shape ``(n, \sum_i c_i)``
        """
        return torch.cat(
            [x.moveaxis(0, -1)[(torch.div(indices, 2 ** i, rounding_mode='trunc')).unbind(1)[1:4]] for i, x in enumerate(feature_pyramid)]
            , dim=1)

def sum_pyramid_channels(base_channels: int, num_scales: int):
    return sum(base_channels * 2 ** i for i in range(num_scales))

class Lambda(nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()

        self.func = func
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs, **self.kwargs)

class SelfSupVoxMMSD(SegmentationAgent):
    """
    The agent for image classificaiton tasks.

    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        The config dictionary should have at least four sections: `dataset`,
        `network`, `training` and `inference`. See :doc:`usage.quickstart` and
        :doc:`usage.fsl` for example.
    """
    def __init__(self, config, stage = 'train'):
        super(SelfSupVoxMMSD, self).__init__(config, stage)
        self.net_dict        = SegNetDict
        self.net_ema = None
        self.writer = SummaryWriter()
        self.criterion = torch.nn.CrossEntropyLoss().to(torch.device('cuda:0'))
        self.transform_dict  = TransformDict
        self.loss_temp = self.config['self_supervised_learning']['loss_temp']
        self.loss_lambda = self.config['self_supervised_learning']['loss_lambda']

        out_dim = self.config['self_supervised_learning']['out_dim']
        warmup_teacher_temp = self.config['self_supervised_learning']['warmup_teacher_temp']
        teacher_temp = self.config['self_supervised_learning']['teacher_temp']
        # use_CRF = self.config['self_supervised_learning']['use_crf']
        warmup_teacher_temp_epochs = self.config['self_supervised_learning']['warmup_teacher_temp_epochs']
        epochs = int(self.config['training']['iter_max']/self.config['training']['iter_valid'])

        self.dino_loss = DINOLoss(
                out_dim,
                2,  # total number of crops = 2 global crops + local_crops_number
                warmup_teacher_temp,
                teacher_temp,
                warmup_teacher_temp_epochs,
                epochs,
            ).cuda()
    

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
    def get_stage_dataset_from_config(self, stage):
        trans_names, trans_params = self.get_transform_names_and_parameters(stage)
        transform_list  = []
        if(trans_names is not None and len(trans_names) > 0):
            for name in trans_names:
                if(name not in self.transform_dict):
                    raise(ValueError("Undefined transform {0:}".format(name))) 
                one_transform = self.transform_dict[name](trans_params)
                transform_list.append(one_transform)
        data_transform = transforms.Compose(transform_list)
        mask_transform = VoxMMSDMaskedImage(trans_params)
        norm_transform = NormalizeWithMinMax(trans_params)

        csv_file  = self.config['dataset'].get(stage + '_csv', None)
        if(stage == 'test'):
            with_label = False 
            self.test_transforms = transform_list
        else:
            with_label = self.config['dataset'].get(stage + '_label', True)
        modal_num = self.config['dataset'].get('modal_num', 1)
        stage_dir = self.config['dataset'].get('train_dir', None)
        if(stage == 'valid' and "valid_dir" in self.config['dataset']):
            stage_dir = self.config['dataset']['valid_dir']
        if(stage == 'test' and "test_dir" in self.config['dataset']):
            stage_dir = self.config['dataset']['test_dir']
        
        dataset  = NiftyDataset(root_dir  = stage_dir,
                                csv_file  = csv_file,
                                modal_num = modal_num,
                                with_label= False,
                                transform = ContrastiveLearningViewGeneratorVoxMMSD(data_transform,mask_transform,norm_transform,
                                                                                    max_num_voxels=self.config['self_supervised_learning']['max_num_voxels'], 
                                                                                    n_views=2), 
                                task = self.task_type)
        return dataset

    def create_network(self):
        proj_dim = self.config['self_supervised_learning']['out_dim']
        super(SelfSupVoxMMSD, self).create_network()
        if(self.net_ema is None):
            net_name = self.config['network']['net_type']
            if(net_name not in SegNetDict):
                raise ValueError("Undefined network {0:}".format(net_name))
            self.net_ema = SegNetDict[net_name](self.config['network'])
        if(self.tensor_type == 'float'):
            self.net_ema.float()
        else:
            self.net_ema.double()

        self.out_conv = nn.Conv3d(32, self.config['network']['class_num'], kernel_size = 1)
        use_bn_in_head = self.config['self_supervised_learning']['use_bn_in_head']
        
        embed_dim = sum_pyramid_channels(base_channels=32,num_scales=4)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
            Lambda(F.normalize)
        )

        self.net, self.net_ema = self.net.cuda(), self.net_ema.cuda()
        self.net_ema.load_state_dict(self.net.state_dict())
        param_number = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        logging.info('parameter number {0:}'.format(param_number))
    
    def create_optimizer(self, params, checkpoint = None):
        """
        Create optimizer based on configuration. 

        :param params: network parameters for optimization. Usually it is obtained by 
            `self.get_parameters_to_update()`.
        """
        opt_params = self.config['training']
        params_groups = get_params_groups(self.net)
        self.optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        epochs = int(opt_params['iter_max']/opt_params['iter_valid'])
        warmup_epochs = int(opt_params['warmup_iter']/opt_params['iter_valid'])
        last_iter = -1
        if(checkpoint is not None):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_iter = checkpoint['iteration'] - 1
        if(self.scheduler is None):
            self.scheduler = cosine_scheduler(
                base_value = 0.001,  # linear scaling rule
                final_value = 0.0001,
                epochs=epochs, 
                niter_per_ep=opt_params['iter_valid'],
                warmup_epochs=warmup_epochs,
            )
            #weight_decay
            self.wd_schedule = cosine_scheduler(
                base_value = 1e-5,
                final_value = 1e-4,
                epochs=epochs, 
                niter_per_ep=opt_params['iter_valid'],
            )
            momentum_teacher = self.config['self_supervised_learning']['momentum_teacher']
            self.momentum_schedule = cosine_scheduler(momentum_teacher, 1,
                                                epochs,
                                                opt_params['iter_valid'])
            
            print(f"Loss, optimizer and schedulers ready.")

    def get_loss_value(self, data, pred, gt, param = None):
        loss_input_dict = {}
        loss_input_dict['prediction'] = pred
        loss_input_dict['ground_truth'] = gt
        loss_value = self.loss_calculator(loss_input_dict)
        return loss_value
    
    def write_scalars(self, train_scalars, lr_value, glob_it):
        loss_scalar ={'train':train_scalars['loss']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        
        logging.info('train loss {0:.4f}, con loss {1:.4f}, rec loss {2:.4f}'.format(
            train_scalars['loss'],train_scalars['con_loss'],train_scalars['rec_loss']) )
    
    def write_image(self, train_images, glob_it):
        input_image=train_images['input_image'][:,47:48,:,:]
        output_image=train_images['output_image'][:,47:48,:,:]
        label_image=train_images['label_image'][:,47:48,:,:]
        
        # input_image=maxmin_normalize(input_image)
        # output_image=maxmin_normalize(output_image)
        # label_image=maxmin_normalize(label_image)
    
        self.summ_writer.add_images('inpput_image',input_image,glob_it)
        self.summ_writer.add_images('output_image',output_image,glob_it)
        self.summ_writer.add_images('label_image',label_image,glob_it)
    
    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        s_feature_pyramid = self.net(patches)
        t_feature_pyramid = self.net_ema(patches)
        rec = self.out_conv(s_feature_pyramid[0])
        s_voxel_feature = torch.cat([select_from_pyramid([x[j] for x in s_feature_pyramid], v) for j, v in enumerate(voxels)])
        t_voxel_feature = torch.cat([select_from_pyramid([x[j] for x in t_feature_pyramid], v) for j, v in enumerate(voxels)])
        
        return s_voxel_feature, t_voxel_feature, rec
    
    def training(self):
        iter_valid  = self.config['training']['iter_valid']
        train_loss  = 0
        train_con_loss  = 0
        train_rec_loss  = 0
        self.net.train()

        scaler = GradScaler(enabled=True)

        for it in range(iter_valid):
            try:
                data = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data = next(self.trainIter)
            # get the inputs
            images = torch.cat(data['image'], dim=0)
            laebls = torch.cat(data['label'], dim=0)
            voxels = torch.cat(data['voxel'], dim=0)

            inputs = self.convert_tensor_type(images)
            labels = self.convert_tensor_type(laebls)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            assert self.net.training
            assert self.proj_head.training

            with amp.autocast(enabled=True):
                s_concat_feature, t_concat_feature, rec = self._vox_to_vec(inputs, voxels)

                embeds_s = self.proj_head(s_concat_feature)
                embeds_t = self.proj_head(t_concat_feature)
                con_loss = self.dino_loss(embeds_s, embeds_t, int(self.glob_it / iter_valid + 1))
                # scaler.scale(con_loss).backward(retain_graph=True)
                rec_loss = self.get_loss_value(data, rec, labels)
                # scaler.scale(rec_loss).backward()
                loss = con_loss + rec_loss
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            scaler.scale(loss).backward()

            scaler.step(self.optimizer)
            scaler.update()

            with torch.no_grad():
                m = self.momentum_schedule[self.glob_it]  # momentum parameter
                for param_q, param_k in zip(self.net.parameters(), self.net_ema.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            train_loss = train_loss + loss.item()
            train_con_loss = train_con_loss + con_loss.item()
            train_rec_loss = train_rec_loss + rec_loss.item()
            
        input_image = inputs[0]
        label_image = labels[0]
        output_image = rec[0]
        train_avg_loss = train_loss / iter_valid
        train_avg_con_loss = train_con_loss / iter_valid
        train_avg_rec_loss = train_rec_loss / iter_valid
        train_scalers = {'loss': train_avg_loss, 'con_loss': train_avg_con_loss, 'rec_loss': train_avg_rec_loss}
        train_images = {'input_image':input_image,'label_image':label_image,'output_image':output_image}
        return train_scalers, train_images

    def train_valid(self):
        device_ids = self.config['training']['gpus']
        if(len(device_ids) > 1):
            self.device = torch.device("cuda:0")
            self.net = nn.DataParallel(self.net, device_ids = device_ids)
        else:
            self.device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net.to(self.device)
        self.proj_head.to(self.device)
        self.out_conv.to(self.device)

        ckpt_dir    = self.config['training']['ckpt_save_dir']
        if(ckpt_dir[-1] == "/"):
            ckpt_dir = ckpt_dir[:-1]
        ckpt_prefix = self.config['training'].get('ckpt_prefix', None)
        if(ckpt_prefix is None):
            ckpt_prefix = ckpt_dir.split('/')[-1]
        # iter_start  = self.config['training']['iter_start']       
        iter_start  = 0     
        iter_max    = self.config['training']['iter_max']
        iter_valid  = self.config['training']['iter_valid']
        iter_save   = self.config['training'].get('iter_save', None)
        early_stop_it = self.config['training'].get('early_stop_patience', None)
        if(iter_save is None):
            iter_save_list = [iter_max]
        elif(isinstance(iter_save, (tuple, list))):
            iter_save_list = iter_save
        else:
            iter_save_list = range(0, iter_max + 1, iter_save)

        self.max_val_dice = 10000
        self.max_val_it   = 0
        self.best_model_wts = None 
        checkpoint = None
        # initialize the network with pre-trained weights
        ckpt_init_name = self.config['training'].get('ckpt_init_name', None)
        ckpt_init_mode = self.config['training'].get('ckpt_init_mode', 0)
        ckpt_for_optm  = None 
        if(ckpt_init_name is not None):
            checkpoint = torch.load(ckpt_dir + "/" + ckpt_init_name, map_location = self.device)
            pretrained_dict = checkpoint['model_state_dict']
            model_dict = self.net.module.state_dict() if (len(device_ids) > 1) else self.net.state_dict()
            if('net1.' in list(model_dict)[0]):
                if(self.config['training']['ckpt_init_mode']>0):
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if \
                    k in model_dict and tensor_shape_match(pretrained_dict[k], model_dict[k])}
                else:
                    pretrained_dict_temp={}
                    for k,v in model_dict.items():
                        if k[5:] in pretrained_dict and tensor_shape_match(pretrained_dict[k[5:]], model_dict[k]):
                            pretrained_dict_temp[k]=pretrained_dict[k[5:]]
                    pretrained_dict=pretrained_dict_temp
                
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if \
                    k in model_dict and tensor_shape_match(pretrained_dict[k], model_dict[k])}
            logging.info("Initializing the following parameters with pre-trained model")
            for k in pretrained_dict:
                logging.info(k)
            if (len(device_ids) > 1):
                self.net.module.load_state_dict(pretrained_dict, strict = False)
            else:
                self.net.load_state_dict(pretrained_dict, strict = False)

            if(ckpt_init_mode > 0): # Load  other information
                self.max_val_dice = checkpoint.get('valid_pred', 0)
                iter_start = checkpoint['iteration']
                self.max_val_it = iter_start
                self.best_model_wts = checkpoint['model_state_dict']
                ckpt_for_optm = checkpoint

        self.create_optimizer(self.get_parameters_to_update(), ckpt_for_optm)
        self.create_loss_calculator()

        self.trainIter  = iter(self.train_loader)
        
        logging.info("{0:} training start".format(str(datetime.now())[:-7]))
        self.summ_writer = SummaryWriter(self.config['training']['ckpt_save_dir'])
        self.glob_it = iter_start
        for it in range(iter_start, iter_max, iter_valid):

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.scheduler[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = self.wd_schedule[it]


            lr_value = self.optimizer.param_groups[0]['lr']
            t0 = time.time()
            train_scalars, train_images = self.training()
            t1 = time.time()
            
            self.glob_it = it + iter_valid
            logging.info("\n{0:} it {1:}".format(str(datetime.now())[:-7], self.glob_it))
            logging.info('learning rate {0:}'.format(lr_value))
            logging.info("training time: {0:.2f}s".format(t1-t0))
            self.write_scalars(train_scalars, lr_value, self.glob_it)
            self.write_image(train_images, self.glob_it)

            if(train_scalars['loss'] < self.max_val_dice):
                self.max_val_dice = train_scalars['loss']
                self.max_val_it   = self.glob_it
                if(len(device_ids) > 1):
                    self.best_model_wts = copy.deepcopy(self.net.module.state_dict())
                else:
                    self.best_model_wts = copy.deepcopy(self.net.state_dict())
                save_dict = {'iteration': self.max_val_it,
                    'valid_pred': self.max_val_dice,
                    'model_state_dict': self.best_model_wts,
                    'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}/{1:}_best.pt".format(ckpt_dir, ckpt_prefix)
                torch.save(save_dict, save_name) 
                txt_file = open("{0:}/{1:}_best.txt".format(ckpt_dir, ckpt_prefix), 'wt')
                txt_file.write(str(self.max_val_it))
                txt_file.close()

            stop_now = True if (early_stop_it is not None and \
                self.glob_it - self.max_val_it > early_stop_it) else False
            if ((self.glob_it in iter_save_list) or stop_now):
                save_dict = {'iteration': self.glob_it,
                             'valid_pred': train_scalars['loss'],
                             'model_state_dict': self.net.module.state_dict() \
                                 if len(device_ids) > 1 else self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}/{1:}_{2:}.pt".format(ckpt_dir, ckpt_prefix, self.glob_it)
                torch.save(save_dict, save_name) 
                txt_file = open("{0:}/{1:}_latest.txt".format(ckpt_dir, ckpt_prefix), 'wt')
                txt_file.write(str(self.glob_it))
                txt_file.close()
            if(stop_now):
                logging.info("The training is early stopped")
                break
        # save the best performing checkpoint
        logging.info('The best performing iter is {0:}, valid dice {1:}'.format(\
            self.max_val_it, self.max_val_dice))
        self.summ_writer.close()

