import numpy as np
import os
import random
import torch
import copy
import torch.nn as nn
from lib.utils.tools import read_pkl
from lib.utils.utils_data import flip_data, crop_scale_3d
    
class Augmenter2D(object):
    """
        Make 2D augmentations on the fly. PyTorch batch-processing GPU version.
    """
    def __init__(self, args):
        self.d2c_params = read_pkl(args.d2c_params_path)
        try:
            self.noise = torch.load(args.noise_path, weights_only=False) # TODO : FIx this it does not work with some pickle ??
        except Exception as e:
            print("Error loading noise: ", e)
            print("Using default noise")
            self.noise = {
                'mean': torch.Tensor([
                        [ 0.0008,  0.0002],
                        [-0.0001,  0.0020],
                        [ 0.0020, -0.0015],
                        [ 0.0009,  0.0009],
                        [ 0.0019, -0.0007],
                        [ 0.0019, -0.0052],
                        [ 0.0009, -0.0019],
                        [ 0.0025,  0.0023],
                        [ 0.0021, -0.0065],
                        [ 0.0017,  0.0017],
                        [ 0.0008,  0.0016],
                        [ 0.0008, -0.0046],
                        [ 0.0015,  0.0003],
                        [ 0.0012,  0.0016],
                        [ 0.0021, -0.0032],
                        [ 0.0017, -0.0003],
                        [ 0.0017,  0.0013]]), 
                'std': torch.Tensor([
                        [0.0037, 0.0036],
                        [0.0071, 0.0051],
                        [0.0061, 0.0050],
                        [0.0090, 0.0053],
                        [0.0067, 0.0053],
                        [0.0070, 0.0049],
                        [0.0095, 0.0056],
                        [0.0066, 0.0050],
                        [0.0055, 0.0053],
                        [0.0053, 0.0049],
                        [0.0034, 0.0035],
                        [0.0080, 0.0061],
                        [0.0052, 0.0048],
                        [0.0050, 0.0051],
                        [0.0078, 0.0079],
                        [0.0060, 0.0057],
                        [0.0054, 0.0057]]), 
                'weight': torch.Tensor([0.9987, 0.9670, 0.8896, 0.8938, 0.9746, 0.8767, 0.8210, 0.9578, 0.9183,
                        0.9679, 0.9841, 0.8910, 0.8016, 0.7028, 0.8809, 0.8246, 0.7287])
                }
            
        self.mask_ratio = args.mask_ratio
        self.mask_T_ratio = args.mask_T_ratio
        self.num_Kframes = 27
        self.noise_std = 0.002

    def dis2conf(self, dis, a, b, m, s):
        f = a/(dis+a)+b*dis
        shift = torch.randn(*dis.shape)*s + m
        # if torch.cuda.is_available():
        shift = shift.to(dis.device)
        return f + shift
    
    def add_noise(self, motion_2d):
        a, b, m, s = self.d2c_params["a"], self.d2c_params["b"], self.d2c_params["m"], self.d2c_params["s"]
        if "uniform_range" in self.noise.keys():
            uniform_range = self.noise["uniform_range"]
        else:
            uniform_range = 0.06
        motion_2d = motion_2d[:,:,:,:2]
        batch_size = motion_2d.shape[0]
        num_frames = motion_2d.shape[1]
        num_joints = motion_2d.shape[2]
        mean = self.noise['mean'].float()
        std = self.noise['std'].float()
        weight = self.noise['weight'][:,None].float()
        sel = torch.rand((batch_size, self.num_Kframes, num_joints, 1))
        gaussian_sample = (torch.randn(batch_size, self.num_Kframes, num_joints, 2) * std + mean) 
        uniform_sample = (torch.rand((batch_size, self.num_Kframes, num_joints, 2))-0.5) * uniform_range
        noise_mean = 0
        delta_noise = torch.randn(num_frames, num_joints, 2) * self.noise_std + noise_mean
        # if torch.cuda.is_available():
        mean = mean.to(motion_2d.device)
        std = std.to(motion_2d.device)
        weight = weight.to(motion_2d.device)
        gaussian_sample = gaussian_sample.to(motion_2d.device)
        uniform_sample = uniform_sample.to(motion_2d.device)
        sel = sel.to(motion_2d.device)
        delta_noise = delta_noise.to(motion_2d.device)
            
        delta = gaussian_sample*(sel<weight) + uniform_sample*(sel>=weight)
        delta_expand = torch.nn.functional.interpolate(delta.unsqueeze(1), [num_frames, num_joints, 2], mode='trilinear', align_corners=True)[:,0]
        delta_final = delta_expand + delta_noise      
        motion_2d = motion_2d + delta_final 
        dx = delta_final[:,:,:,0]
        dy = delta_final[:,:,:,1]
        dis2 = dx*dx+dy*dy
        dis = torch.sqrt(dis2)
        conf = self.dis2conf(dis, a, b, m, s).clip(0,1).reshape([batch_size, num_frames, num_joints, -1])
        return torch.cat((motion_2d, conf), dim=3)
        
    def add_mask(self, x):
        ''' motion_2d: (N,T,17,3)
        '''
        N,T,J,C = x.shape
        mask = torch.rand(N,T,J,1, dtype=x.dtype, device=x.device) > self.mask_ratio
        mask_T = torch.rand(1,T,1,1, dtype=x.dtype, device=x.device) > self.mask_T_ratio
        x = x * mask * mask_T
        return x
    
    def augment2D(self, motion_2d, mask=False, noise=False):     
        if noise:
            motion_2d = self.add_noise(motion_2d)
        if mask:
            motion_2d = self.add_mask(motion_2d)
        return motion_2d
    
class Augmenter3D(object):
    """
        Make 3D augmentations when dataloaders get items. NumPy single motion version.
    """
    def __init__(self, args):
        self.flip = args.flip
        if hasattr(args, "scale_range_pretrain"):
            self.scale_range_pretrain = args.scale_range_pretrain
        else:
            self.scale_range_pretrain = None
    
    def augment3D(self, motion_3d):
        if self.scale_range_pretrain:
            motion_3d = crop_scale_3d(motion_3d, self.scale_range_pretrain)
        if self.flip and random.random()>0.5:                       
            motion_3d = flip_data(motion_3d)
        return motion_3d