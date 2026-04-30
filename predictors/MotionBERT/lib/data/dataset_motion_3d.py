import torch
import numpy as np
import glob
import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, "..", ".."))
import io
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from lib.data.augmentation import Augmenter3D
from lib.utils.tools import read_pkl
from lib.utils.utils_data import flip_data
    
class MotionDataset(Dataset):
    def __init__(self, args, subset_list, data_split): # data_split: train/test
        np.random.seed(0)
        self.data_root = args.data_root
        self.subset_list = subset_list
        self.data_split = data_split
        file_list_all = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list_all.append(os.path.join(data_path, i))
        self.file_list = file_list_all
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 
    
class SimpleMotionDataset(Dataset):
    def __init__(self, data_root, subset_list, data_split): # data_split: train/test
        np.random.seed(0)
        self.data_root = data_root
        self.subset_list = subset_list
        self.data_split = data_split
        file_list_all = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list_all.append(os.path.join(data_path, i))
        self.file_list = file_list_all
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 

class MotionDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(MotionDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["data_label"]  
        
        if self.data_split=="train":
            if self.synthetic or self.gt_2d:
                motion_3d = self.aug.augment3D(motion_3d)
                motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1                        # No 2D detection, use GT xy and c=1.
            elif motion_file["data_input"] is not None:     # Have 2D detection 
                motion_2d = motion_file["data_input"]
                if self.flip and random.random() > 0.5:                        # Training augmentation - random flipping
                    motion_2d = flip_data(motion_2d)
                    motion_3d = flip_data(motion_3d)
            else:
                raise ValueError('Training illegal.') 
            
        elif self.data_split=="test":                                           
            motion_2d = motion_file["data_input"]
            if self.gt_2d:
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1
                
        else:
            raise ValueError('Data split unknown.')
            
        return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)
class SimpleMotionDataset3D(SimpleMotionDataset):
    def __init__(self, data_root, subset_list, data_split, flip = True, n_frames=48, subsample_factor=1):
        super(SimpleMotionDataset3D, self).__init__(data_root, subset_list, data_split)
        self.flip = flip
        self.n_frames = n_frames
        self.subsample_factor = subsample_factor
        
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        # print(f"Processing {file_path}")
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["data_label"]  
        
        fake_label = torch.tensor(0)
        dataset_name = "AMASS" if "AMASS" in file_path else "H36M-SH" if "H36M-SH" in file_path else "None"
        metadata = {
            "unique_track_identifier": f"{dataset_name}_{file_path.split('/')[-1].split('.')[0]}",
            "recording": dataset_name,
            "flipped": False,
            "file_path": file_path,
            "orig_dataset": dataset_name,
        }

        motion_3d = motion_3d[:self.n_frames*self.subsample_factor]
        motion_3d = motion_3d[::self.subsample_factor]
        if self.flip and random.random() > 0.5:                   # Training augmentation - random flipping
            motion_3d = flip_data(motion_3d)
            metadata["flipped"] = True

        return torch.FloatTensor(motion_3d), fake_label, metadata # torch.FloatTensor(motion_3d)