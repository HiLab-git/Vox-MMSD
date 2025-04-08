# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os 
import torch
import torch.nn as nn
import random
import numpy as np 
import math
import warnings
from scipy import ndimage
from pymic.io.image_read_write import *
from pymic.util.image_process import *


def get_human_region_mask(img):
    """
    Get the mask of human region in CT volumes
    """
    dim = len(img.shape)
    if( dim == 4):
        img = img[0]
    mask = np.asarray(img > -600)
    se = np.ones([3,3,3])
    mask = ndimage.binary_opening(mask, se, iterations = 2)
    D, H, W = mask.shape 
    for h in range(H):
        if(mask[:,h,:].sum() < 2000):
            mask[:,h, :] = np.zeros((D, W))
    mask = get_largest_k_components(mask, 1)
    mask_close = ndimage.binary_closing(mask, se, iterations = 2)

    D, H, W = mask.shape
    for d in [1, 2, D-3, D-2]:
        mask_close[d] = mask[d]
    for d in range(0, D, 2):
        mask_close[d, 2:-2, 2:-2] = np.ones((H-4, W-4))
    
    # get background component
    bg = np.zeros_like(mask)
    bgs = get_largest_k_components(1- mask_close, 10)
    for bgi in bgs:
        indices = np.where(bgi)
        if(bgi.sum() < 1000):
            break
        if(indices[0].min() == 0 or indices[1].min() == 0 or indices[2].min() ==0 or \
           indices[0].max() == D-1 or indices[1].max() == H-1 or indices[2].max() ==W-1):
            bg = bg + bgi
    fg = 1 - bg 

    fg = ndimage.binary_opening(fg, se, iterations = 1)
    fg = get_largest_k_components(fg, 1)
    if(dim == 4):
        fg = np.expand_dims(fg, 0)
    fg = np.asarray(fg, np.uint8)
    return fg 

def get_human_region_mask_fast(img, itk_spacing):
    # downsample
    D, H, W = img.shape 
    # scale_down = [1, 1, 1]
    if(itk_spacing[2] <= 1):
        scale_down = [1/2, 1/2, 1/2]
    else:
        scale_down = [1, 1/2, 1/2]
    img_sub    = ndimage.interpolation.zoom(img, scale_down, order = 0)
    mask       = get_human_region_mask(img_sub)
    D1, H1, W1 = mask.shape 
    scale_up = [D/D1, H/H1, W/W1]
    mask = ndimage.interpolation.zoom(mask, scale_up, order = 0)
    return mask

def crop_ct_scan(input_img, output_img, input_lab = None, output_lab = None, z_axis_density = 0.5):
    """
    Crop a CT scan based on the bounding box of the human region. 
    """
    img_obj = sitk.ReadImage(input_img)
    img     = sitk.GetArrayFromImage(img_obj)
    mask    = np.asarray(img > -600)
    mask2d  = np.mean(mask, axis = 0) > z_axis_density
    se      = np.ones([3,3])
    mask2d  = ndimage.binary_opening(mask2d, se, iterations = 2)
    mask2d  = get_largest_k_components(mask2d, 1)
    bbmin, bbmax = get_ND_bounding_box(mask2d, margin = [0, 0])
    bbmin   = [0] + bbmin
    bbmax   = [img.shape[0]] + bbmax
    img_sub = crop_ND_volume_with_bounding_box(img, bbmin, bbmax)
    img_sub_obj = sitk.GetImageFromArray(img_sub)
    img_sub_obj.SetSpacing(img_obj.GetSpacing())
    img_sub_obj.SetDirection(img_obj.GetDirection())
    sitk.WriteImage(img_sub_obj, output_img)
    if(input_lab is not None):
        lab_obj  = sitk.ReadImage(input_lab)
        lab = sitk.GetArrayFromImage(lab_obj)
        lab_sub = crop_ND_volume_with_bounding_box(lab, bbmin, bbmax)
        lab_sub_obj = sitk.GetImageFromArray(lab_sub)
        lab_sub_obj.SetSpacing(img_obj.GetSpacing())
        sitk.WriteImage(lab_sub_obj, output_lab)

def get_human_body_mask_and_crop(input_dir, out_img_dir, out_mask_dir):
    if(not os.path.exists(out_img_dir)):
        os.mkdir(out_img_dir)
        os.mkdir(out_mask_dir)

    img_names = [item for item in os.listdir(input_dir) if "nii.gz" in item]
    img_names  = sorted(img_names)
    for img_name in img_names:
        print(img_name)
        input_name = input_dir + "/" + img_name
        out_name   = out_img_dir + "/" + img_name 
        mask_name  = out_mask_dir + "/" + img_name 
        if(os.path.isfile(out_name)):
            continue
        img_obj = sitk.ReadImage(input_name)
        img     = sitk.GetArrayFromImage(img_obj)
        spacing = img_obj.GetSpacing()

        # downsample
        D, H, W = img.shape 
        spacing = img_obj.GetSpacing()
        # scale_down = [1, 1, 1]
        if(spacing[2] <= 1):
            scale_down = [1/2, 1/2, 1/2]
        else:
            scale_down = [1, 1/2, 1/2]
        img_sub    = ndimage.interpolation.zoom(img, scale_down, order = 0)
        mask       = get_human_region_mask(img_sub)
        D1, H1, W1 = mask.shape 
        scale_up = [D/D1, H/H1, W/W1]
        mask = ndimage.interpolation.zoom(mask, scale_up, order = 0)

        bbmin, bbmax = get_ND_bounding_box(mask)
        img_crop  = crop_ND_volume_with_bounding_box(img, bbmin, bbmax)
        mask_crop = crop_ND_volume_with_bounding_box(mask, bbmin, bbmax)

        out_img_obj = sitk.GetImageFromArray(img_crop)
        out_img_obj.SetSpacing(spacing)
        sitk.WriteImage(out_img_obj, out_name)
        mask_obj = sitk.GetImageFromArray(mask_crop)
        mask_obj.CopyInformation(out_img_obj)
        sitk.WriteImage(mask_obj, mask_name)

def volume_fusion(x, fg_num, block_range, size_min, size_max):
    """
    Fuse a subregion of an impage with another one to generate
    images and labels for self-supervised segmentation.
    input x should be a batch of tensors
    """
    #n_min, n_max,  
    N, C, D, H, W = list(x.shape)
    fg_mask = torch.zeros_like(x).to(torch.int32)
    # generate mask 
    for n in range(N):
        p_num = random.randint(block_range[0], block_range[1])
        for i in range(p_num):
            d = random.randint(size_min[0], size_max[0])
            h = random.randint(size_min[1], size_max[1])
            w = random.randint(size_min[2], size_max[2])
            dc = random.randint(0, D - 1)
            hc = random.randint(0, H - 1)
            wc = random.randint(0, W - 1)
            d0 = dc - d // 2
            h0 = hc - h // 2
            w0 = wc - w // 2
            d1 = min(D, d0 + d)
            h1 = min(H, h0 + h)
            w1 = min(W, w0 + w)
            d0, h0, w0 = max(0, d0), max(0, h0), max(0, w0) 
            temp_m = torch.ones([C, d1 - d0, h1 - h0, w1 - w0]) * random.randint(1, fg_num)
            fg_mask[n, :, d0:d1, h0:h1, w0:w1] = temp_m
    fg_w   = fg_mask * 1.0 / fg_num
    x_roll = torch.roll(x, 1, 0)
    x_fuse = fg_w*x_roll + (1.0 - fg_w)*x     
    # y_prob = get_one_hot_seg(fg_mask.to(torch.int32), fg_num + 1)
    return x_fuse, fg_mask 

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DINOHead_modmask(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        
        #cls output和modmask output共享head
        # self.mlp2 = self.mlp
        self.last_layer2 = self.last_layer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x1 = self.last_layer(x)
        x2 = self.last_layer2(x)
        return x1, x2
    

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            encoder_out, decoder_out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[-1]
            # accumulate outputs
            output = torch.cat((output, encoder_out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output), decoder_out

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None
