import numpy as np
import torch
from PIL import Image 
from pymic.transform.intensity import NonLinearTransform
from pymic.transform.normalize import NormalizeWithMeanStd

import copy
import random
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
from scipy.ndimage import zoom
from imops import crop_to_box
np.random.seed(0)

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def sample_box(image_size, patch_size, anchor_voxel=None):
    image_size = np.array(image_size, ndmin=1)
    patch_size = np.array(patch_size, ndmin=1)

    if not np.all(image_size >= patch_size):
        raise ValueError(f'Can\'t sample patch of size {patch_size} from image of size {image_size}')

    min_start = 0
    max_start = image_size - patch_size
    if anchor_voxel is not None:
        anchor_voxel = np.array(anchor_voxel, ndmin=1)
        min_start = np.maximum(min_start, anchor_voxel - patch_size + 1)
        max_start = np.minimum(max_start, anchor_voxel)
    start = np.random.randint(min_start, max_start + 1)
    return np.array([start, start + patch_size])

class ContrastiveLearningViewGeneratorVoxMMSD(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, mask_transform, norm_transform, max_num_voxels,  n_views=2):
        self.base_transform = base_transform
        self.mask_transform = mask_transform
        self.n_views = n_views
        self.max_num_voxels = max_num_voxels
        self.norm = norm_transform

    def __call__(self, x):
        patch_size = [4, 96, 96, 96]
        max_num_voxels = self.max_num_voxels
        while(1):
            sample_1 = copy.deepcopy(x)
            sample_2 = copy.deepcopy(x)
            image_1 = sample_1['image']
            image_2 = sample_2['image']
            voxels_1 = np.argwhere(image_1!=0)
            voxels_2 = np.argwhere(image_2!=0)
            # print(voxels_1.max(),voxels_2.max())

            box_1 = sample_box(image_1.shape, patch_size)
            box_2 = sample_box(image_2.shape, patch_size)
            image_1 = image_1[tuple(slice(st,end) for st,end in zip(box_1[0],box_1[1]))]
            image_2 = image_2[tuple(slice(st,end) for st,end in zip(box_2[0],box_2[1]))]
            sample_1['image'] = image_1
            sample_2['image'] = image_2
            label_1 = copy.deepcopy(self.norm(sample_1)['image'])
            label_2 = copy.deepcopy(self.norm(sample_2)['image'])
            image_1 = self.mask_transform(sample_1)['image']
            image_2 = self.mask_transform(sample_2)['image']

            shift_1 = box_1[0]
            voxels_1 = voxels_1 - shift_1 
            shift_2 = box_2[0]
            voxels_2 = voxels_2 - shift_2
            
            valid_1 = np.all((voxels_1 >= 0) & (voxels_1 < patch_size), axis=1)
            valid_2 = np.all((voxels_2 >= 0) & (voxels_2 < patch_size), axis=1)
            valid = valid_1 & valid_2
            indices = np.where(valid)[0]

            overlapping_voxels_1 = voxels_1[indices]
            overlapping_voxels_2 = voxels_2[indices]
            # print(overlapping_voxels_1.max(),overlapping_voxels_2.max())
            random_mod_index_1 = np.random.randint(0,4,size=overlapping_voxels_1.shape[0])
            random_mod_index_2 = np.random.randint(0,4,size=overlapping_voxels_2.shape[0])
            unmasked_valid_1 = image_1[random_mod_index_1,overlapping_voxels_1[:,1],overlapping_voxels_1[:,2],overlapping_voxels_1[:,3]]>0
            unmasked_valid_2 = image_2[random_mod_index_2,overlapping_voxels_2[:,1],overlapping_voxels_2[:,2],overlapping_voxels_2[:,3]]>0

            unmasked_valid = unmasked_valid_1 & unmasked_valid_2
            
            if(np.sum(unmasked_valid)>=max_num_voxels):
                choice_index = np.random.choice(np.arange(unmasked_valid.shape[0])[unmasked_valid],max_num_voxels, replace=False)
                final_indices_1 = np.array([random_mod_index_1[choice_index],overlapping_voxels_1[choice_index,1],overlapping_voxels_1[choice_index,2],overlapping_voxels_1[choice_index,3]]).T
                final_indices_2 = np.array([random_mod_index_2[choice_index],overlapping_voxels_2[choice_index,1],overlapping_voxels_2[choice_index,2],overlapping_voxels_2[choice_index,3]]).T
                break

        sample_1['image']=image_1
        sample_2['image']=image_2 
        image_1 = self.base_transform(sample_1)['image']
        image_2 = self.base_transform(sample_2)['image']

        x['label'] = [label_1, label_2]
        x['image'] = [image_1, image_2]
        x['voxel'] = [final_indices_1, final_indices_2]
        return x