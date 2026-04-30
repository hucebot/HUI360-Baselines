import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import pickle
# import cPickle as pkl
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, "..", ".."))
import random
import copy
import json
from collections import defaultdict
from lib.utils.utils_data import crop_scale, flip_data, resample, split_clips


def jrdb2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        JRDB: 
            0- head
            1- right eye
            2- left eye
            3- right shoulder
            4- center shoulder
            5- left shoulder
            6- right elbow
            7- left elbow
            8- center hip
            9- right wrist
            10- right hip
            11- left hip
            12- left wrist
            13- right knee
            14- left knee
            15- right foot
            16- left foot
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,:,0,:] = (x[:,:,10,:] + x[:,:,11,:]) * 0.5
    y[:,:,1,:] = x[:,:,10,:]
    y[:,:,2,:] = x[:,:,13,:]
    y[:,:,3,:] = x[:,:,15,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,14,:]
    y[:,:,6,:] = x[:,:,16,:]
    y[:,:,8,:] = (x[:,:,3,:] + x[:,:,5,:]) * 0.5
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
    y[:,:,10,:] = x[:,:,0,:]
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,7,:]
    y[:,:,13,:] = x[:,:,12,:]
    y[:,:,14,:] = x[:,:,3,:]
    y[:,:,15,:] = x[:,:,6,:]
    y[:,:,16,:] = x[:,:,9,:]
    return y
    
    
def posetrack2h36m(x):
    '''
        Input: x (T x V x C)

        PoseTrack keypoints = [ 'nose',
                                'head_bottom',
                                'head_top',
                                'left_ear',
                                'right_ear',
                                'left_shoulder',
                                'right_shoulder',
                                'left_elbow',
                                'right_elbow',
                                'left_wrist',
                                'right_wrist',
                                'left_hip',
                                'right_hip',
                                'left_knee',
                                'right_knee',
                                'left_ankle',
                                'right_ankle']
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,0,:] = (x[:,11,:] + x[:,12,:]) * 0.5
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,8,:] = x[:,1,:]
    y[:,7,:] = (y[:,0,:] + y[:,8,:]) * 0.5
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,2,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    y[:,0,2] = np.minimum(x[:,11,2], x[:,12,2])
    y[:,7,2] = np.minimum(y[:,0,2], y[:,8,2])
    return y


# COCO18 format
# ['Nose', 'Neck', 'RSho', 'RElb', 'RWr', 'LSho', 'LElb', 'LWr', 'RHip', 'RKnee', 'RAnk', 'LHip', 'LKnee', 'LAnk', 'REye', 'LEye', 'REar', 'LEar']

# Openpose format
# ['Nose', 'Neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri','Rhip',  'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

#  H36M format
# ['root', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 'lank', 'belly', 'neck', 'nose', 'head', 'lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri']


def openpose2h36m(x):
    '''
        Input: x (..., 18, C) -> output (..., 17, C)

        OpenPose: ['Nose', 'Neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
                   'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

        H36M: ['root', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 'lank', 'belly', 'neck', 'nose',
               'head', 'lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri']
    '''
    out_shape = list(x.shape)
    out_shape[-2] = 17
    y = np.zeros(out_shape, dtype=x.dtype)
    y[..., 0, :] = (x[..., 8, :] + x[..., 11, :]) * 0.5   # root
    y[..., 1, :] = x[..., 8, :]   # rhip
    y[..., 2, :] = x[..., 9, :]   # rkne
    y[..., 3, :] = x[..., 10, :]  # rank
    y[..., 4, :] = x[..., 11, :]  # lhip
    y[..., 5, :] = x[..., 12, :]  # lkne
    y[..., 6, :] = x[..., 13, :]  # lank
    y[..., 8, :] = x[..., 1, :]   # neck
    y[..., 7, :] = (y[..., 0, :] + y[..., 8, :]) * 0.5   # belly
    y[..., 9, :] = x[..., 0, :]   # nose
    y[..., 10, :] = (x[..., 14, :] + x[..., 15, :]) * 0.5  # head (Leye, Reye)
    y[..., 11, :] = x[..., 5, :]  # lsho
    y[..., 12, :] = x[..., 6, :]  # lelb
    y[..., 13, :] = x[..., 7, :]  # lwri
    y[..., 14, :] = x[..., 2, :]  # rsho
    y[..., 15, :] = x[..., 3, :]  # relb
    y[..., 16, :] = x[..., 4, :]  # rwri
    return y

def coco182h36m(x):
    '''
        Input: x (..., 18, C) -> output (..., 17, C)

        COCO18: ['Nose', 'Neck', 'RSho', 'RElb', 'RWr', 'LSho', 'LElb', 'LWr',
                 'RHip', 'RKnee', 'RAnk', 'LHip', 'LKnee', 'LAnk', 'REye', 'LEye', 'REar', 'LEar']

        H36M: ['root', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 'lank', 'belly', 'neck', 'nose',
               'head', 'lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri']
    '''
    out_shape = list(x.shape)
    out_shape[-2] = 17
    y = np.zeros(out_shape, dtype=x.dtype)
    y[..., 0, :] = (x[..., 8, :] + x[..., 11, :]) * 0.5   # root (hip center)
    y[..., 1, :] = x[..., 8, :]   # rhip
    y[..., 2, :] = x[..., 9, :]  # rkne
    y[..., 3, :] = x[..., 10, :] # rank
    y[..., 4, :] = x[..., 11, :] # lhip
    y[..., 5, :] = x[..., 12, :] # lkne
    y[..., 6, :] = x[..., 13, :] # lank
    y[..., 8, :] = x[..., 1, :]  # neck
    y[..., 7, :] = (y[..., 0, :] + y[..., 8, :]) * 0.5   # belly
    y[..., 9, :] = x[..., 0, :]  # nose
    y[..., 10, :] = (x[..., 14, :] + x[..., 15, :]) * 0.5  # head (REye, LEye)
    y[..., 11, :] = x[..., 5, :]  # lsho
    y[..., 12, :] = x[..., 6, :]  # lelb
    y[..., 13, :] = x[..., 7, :]  # lwri
    y[..., 14, :] = x[..., 2, :]  # rsho
    y[..., 15, :] = x[..., 3, :]  # relb
    y[..., 16, :] = x[..., 4, :]  # rwri
    return y
 
def coco2h36m(x):
    '''
        Input: x (M x T x V x C) or (B, T, V, C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
    y[:,:,1,:] = x[:,:,12,:]
    y[:,:,2,:] = x[:,:,14,:]
    y[:,:,3,:] = x[:,:,16,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,13,:]
    y[:,:,6,:] = x[:,:,15,:]
    y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = x[:,:,0,:]
    y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,7,:]
    y[:,:,13,:] = x[:,:,9,:]
    y[:,:,14,:] = x[:,:,6,:]
    y[:,:,15,:] = x[:,:,8,:]
    y[:,:,16,:] = x[:,:,10,:]
    return y

class PoseTrackDataset2D(Dataset):
    def __init__(self, data_root, split, flip=True, scale_range=[0.25, 1], n_frames=48):
        super(PoseTrackDataset2D, self).__init__()
        self.flip = flip
        self.n_frames = n_frames
        file_list = sorted(os.listdir(os.path.join(data_root, split)))
        file_list = [filename for filename in file_list if filename.endswith('.json')]
        all_motions = []
        all_motions_filtered = []
        self.scale_range = scale_range
        
        for filename in file_list:
            with open(os.path.join(data_root, split, filename), 'r') as file:
                json_dict = json.load(file)
                annots = json_dict['annotations']
                imgs = json_dict['images']
                motions = defaultdict(list)
                for annot in annots:
                    tid = annot['track_id']
                    pose2d = np.array(annot['keypoints']).reshape(-1,3)
                    motions[tid].append(pose2d)
            all_motions += list(motions.values())
            
        not_long_enough_counter = 0
        invalid_counter = 0
        for motion in all_motions:
            if len(motion)<n_frames:
                not_long_enough_counter += 1
                continue
            motion = np.array(motion[:n_frames])
            if np.sum(motion[:,:,2]) <= n_frames*9:  # Valid joint num threshold
                invalid_counter += 1
                continue
            motion = crop_scale(motion, self.scale_range) 
            motion = posetrack2h36m(motion)
            motion[motion[:,:,2]==0] = 0
            if np.sum(motion[:,0,2]) < n_frames:
                invalid_counter += 1
                continue                      # Root all visible (needed for framewise rootrel)
            all_motions_filtered.append(motion)
        all_motions_filtered = np.array(all_motions_filtered)
        
        N = all_motions_filtered.shape[0]
        if N > 0:
            all_motions_filtered_tensor = torch.FloatTensor(all_motions_filtered)
            invalid_keypoints_mask = all_motions_filtered_tensor[...,2] < 0.25 # shape (N, T, V)
            input_tensor_for_min = all_motions_filtered_tensor[...,:2].clone()
            input_tensor_for_min[invalid_keypoints_mask] = float('inf')
            input_tensor_for_max = all_motions_filtered_tensor[...,:2].clone()
            input_tensor_for_max[invalid_keypoints_mask] = float('-inf')
            min_xy_per_sample = input_tensor_for_min[...,:2].amin(dim=2) # shape (N, T, 2)
            max_xy_per_sample = input_tensor_for_max[...,:2].amax(dim=2) # shape (N, T, 2)
            range_xy_per_sample = max_xy_per_sample - min_xy_per_sample # shape (N, T, 2)
            invalid_range_mask = range_xy_per_sample < 0.01 # shape (N, T, 2)
            invalid_range_mask = invalid_range_mask.numpy().astype(bool) # shape (N, T, 2)
            invalid_range_mask = np.any(invalid_range_mask, axis=(1,2)) # shape (N,)
            all_motions_filtered = all_motions_filtered[~invalid_range_mask] # shape (N, T, 17, 3)
        else:
            invalid_range_mask = np.ones(N, dtype=bool)
        
        self.motions_2d = all_motions_filtered # shape (N, T, 17, 3)

        print(f"[PoseTrackDataset2D] : kept {len(self.motions_2d)} out of {len(all_motions)} motions (not long enough: {not_long_enough_counter}, invalid: {invalid_counter}, invalid range: {invalid_range_mask.sum()})")
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions_2d)

    def __getitem__(self, index):
        'Generates one sample of data'
        metadata = {
            "unique_track_identifier": f"PoseTrack_{index}",
            "recording": "PoseTrack",
            "flipped": False,
            "file_path": "motion_all.npy",
            "orig_dataset": "PoseTrack18",
        }
        fake_label = torch.tensor(0)
        motion_2d = torch.FloatTensor(self.motions_2d[index])
        if self.flip and random.random()>0.5:                       
            motion_2d = flip_data(motion_2d)
            metadata["flipped"] = True
        return motion_2d, fake_label, metadata # motion_2d
    
class InstaVDataset2D(Dataset):
    def __init__(self, data_root, split, n_frames=81, data_stride=27, flip=True, valid_threshold=0.0, min_valid_joints=9, scale_range=[1, 1]):
        super(InstaVDataset2D, self).__init__()
        self.flip = flip
        self.scale_range = scale_range
        self.n_frames = n_frames
        motion_all = np.load(os.path.join(data_root, 'motion_all.npy'))
        id_all = np.load(os.path.join(data_root, 'id_all.npy'))
        split_id = split_clips(id_all, n_frames, data_stride)       # list of ranges of indexes
        motions_2d = motion_all[split_id]                           # [N, T, 17, 3]
        
        valid_joints = motions_2d[:,:,:,2] > valid_threshold # [N, T, 17]
        valid_joints_count = np.sum(valid_joints, axis=2) # [N, T]
        valid_pose = valid_joints_count >= min_valid_joints # [N, T]
        all_valid_poses = np.all(valid_pose, axis=1) # [N]
        all_motions_filtered = motions_2d[all_valid_poses, ...]

        N = all_motions_filtered.shape[0]
        if N > 0:
            all_motions_filtered_tensor = torch.FloatTensor(all_motions_filtered)
            invalid_keypoints_mask = all_motions_filtered_tensor[...,2] < 0.25 # shape (N, T, V)
            input_tensor_for_min = all_motions_filtered_tensor[...,:2].clone()
            input_tensor_for_min[invalid_keypoints_mask] = float('inf')
            input_tensor_for_max = all_motions_filtered_tensor[...,:2].clone()
            input_tensor_for_max[invalid_keypoints_mask] = float('-inf')
            min_xy_per_sample = input_tensor_for_min[...,:2].amin(dim=2) # shape (N, T, 2)
            max_xy_per_sample = input_tensor_for_max[...,:2].amax(dim=2) # shape (N, T, 2)
            range_xy_per_sample = max_xy_per_sample - min_xy_per_sample # shape (N, T, 2)
            invalid_range_mask = range_xy_per_sample < 0.01 # shape (N, T, 2)
            invalid_range_mask = invalid_range_mask.numpy().astype(bool) # shape (N, T, 2)
            invalid_range_mask = np.any(invalid_range_mask, axis=(1,2)) # shape (N,)
            all_motions_filtered = all_motions_filtered[~invalid_range_mask] # shape (N, T, 17, 3)
        else:
            invalid_range_mask = np.ones(N, dtype=bool)
            
        self.motions_2d = all_motions_filtered

        print(f"[InstaVDataset2D] : kept {len(self.motions_2d)} out of {len(split_id)} motions (unique ids : {len(np.unique(id_all))}, invalid range: {invalid_range_mask.sum()})")
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions_2d)

    def __getitem__(self, index):
        'Generates one sample of data'
        metadata = {
            "unique_track_identifier": f"InstaVariety_{index}",
            "recording": "InstaVariety",
            "flipped": False,
            "file_path": "motion_all.npy",
            "orig_dataset": "InstaVariety",
        }
        fake_label = torch.tensor(0)
        motion_2d = self.motions_2d[index]
        motion_2d = crop_scale(motion_2d, self.scale_range) 
        motion_2d[motion_2d[:,:,2]==0] = 0
        
        # fix belly, as the middle of root and neck
        motion_2d[:,7,:] = (motion_2d[:,0,:] + motion_2d[:,8,:]) * 0.5
        
        if self.flip and random.random()>0.5:                       
            motion_2d = flip_data(motion_2d)
            metadata["flipped"] = True
            
        motion_2d = torch.FloatTensor(motion_2d)
        return motion_2d, fake_label, metadata # motion_2d
    
    
# predictors/MotionBERT/data/JRDB/train/labels/labels_2d_pose_stitched_coco
class JRDBDataset2D(Dataset):
    def __init__(self, data_root, split, n_frames=48, data_stride=48, valid_threshold=0.0, min_valid_joints=9, flip=True):
        
        super(JRDBDataset2D, self).__init__()
        self.n_frames = n_frames
        self.data_root = data_root
        self.flip = flip
        
        validation_files_places = ["cubberly", "hewlett", "memorial"]
        all_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(os.path.join(self.data_root))) for f in fn if f.endswith(".json")]
        all_files.sort()
        
        validation_files = []
        training_files = []
        for file in all_files:
            if any(place in file for place in validation_files_places):
                validation_files.append(file)
            else:
                training_files.append(file)
        
        if split == "train":
            self.files = training_files
        elif split == "val":
            self.files = validation_files
        else:
            raise ValueError(f"Invalid split: {split}")
                
        total_tracks = 0
        total_segments = 0
        motions_2d = []
        for file in self.files:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # load list of individual images infos
            images_infos = data["images"]
            images_count = len(images_infos) # not that images id starts at 1
            
            # load pose annotations 
            annotations = data["annotations"]
            
            # count the number of annotations by track id
            tracks_annotations = {}
            for annot_iter, annot in enumerate(annotations):
                track_id = annot["track_id"]
                if track_id not in tracks_annotations:
                    tracks_annotations[track_id] = {"count": 0, "image_ids": [], "annot_iter": []}
                tracks_annotations[track_id]["count"] += 1
                tracks_annotations[track_id]["image_ids"].append(annot["image_id"]-1) # image id starts at 1
                tracks_annotations[track_id]["annot_iter"].append(annot_iter)
            
            valid_segments = {}
            for track_id, track_info in tracks_annotations.items():
                if track_info["count"] >= n_frames:
                    # check that the indexes are consecutive
                    track_images_ids_array = np.array(track_info["image_ids"])
                    track_images_ids_argsort = np.argsort(track_images_ids_array)
                    
                    track_images_ids_array = track_images_ids_array[track_images_ids_argsort]
                    track_annot_iters_array = np.array(track_info["annot_iter"])[track_images_ids_argsort]
                    
                    track_id_segment_counter = 0
                    # search for one valid segment of n_frames consecutive indexes
                    for i in range(0, len(track_images_ids_array) - n_frames + 1, data_stride):
                        if np.all(np.diff(track_images_ids_array[i:i+n_frames]) == 1):
                            # consecutive images ids
                            segment_id = track_id * 1000 + track_id_segment_counter
                            valid_segments[segment_id] = {"count": n_frames, "image_ids": track_images_ids_array[i:i+n_frames], "annot_iters": track_annot_iters_array[i:i+n_frames]}
                            track_id_segment_counter += 1
                    else:
                        continue
                else:
                    continue
                
            for segment_id, segment_info in valid_segments.items():
                annot_ids = segment_info["annot_iters"]
                motion_segment = []
                for annot_id in annot_ids:
                    annot = annotations[annot_id]
                    keypoints = annot["keypoints"]
                    keypoints = np.array(keypoints).reshape(-1,3)
                    motion_segment.append(keypoints)
                
                motion_segment = np.array(motion_segment) # [T, V, 3]
                
                # check enough valid keypoints per skeleton
                scores = motion_segment[:,:,2] # [T, V]
                valid_keypoints_mask = scores > valid_threshold # [T, V], score of 0 = invisible, 1 = occluded, 2 = visible
                valid_keypoints_count = np.sum(valid_keypoints_mask, axis=1) # [T]
                valid_keypoints_count_mask = valid_keypoints_count >= min_valid_joints # [T], at least 9 valid keypoints per skeleton
                if np.all(valid_keypoints_count_mask):
                    
                    # check if wrap around
                    xs = motion_segment[:,:,0] # [T, V]
                    if xs.max() - xs.min() > 3760/2:
                        # left keypoints mask
                        xs_left_mask = xs < 3760/2 # [T, V]
                        motion_segment[xs_left_mask,0] = motion_segment[xs_left_mask,0] + 3760 # move to the right
                        # recenter
                        motion_segment[:,:,0] = motion_segment[:,:,0] - 3760/2
                    
                    motions_2d.append(motion_segment)
                else:
                    continue
                                
            total_tracks += len(tracks_annotations)
            total_segments += len(valid_segments)
        
        all_motions_filtered = np.array(motions_2d)
        all_motions_filtered = jrdb2h36m(all_motions_filtered) # converts the JRDB keypoints to the H36M keypoints

        N = all_motions_filtered.shape[0]
        if N > 0:
            all_motions_filtered_tensor = torch.FloatTensor(all_motions_filtered)
            invalid_keypoints_mask = all_motions_filtered_tensor[...,2] < 0.25 # shape (N, T, V)
            input_tensor_for_min = all_motions_filtered_tensor[...,:2].clone()
            input_tensor_for_min[invalid_keypoints_mask] = float('inf')
            input_tensor_for_max = all_motions_filtered_tensor[...,:2].clone()
            input_tensor_for_max[invalid_keypoints_mask] = float('-inf')
            min_xy_per_sample = input_tensor_for_min[...,:2].amin(dim=2) # shape (N, T, 2)
            max_xy_per_sample = input_tensor_for_max[...,:2].amax(dim=2) # shape (N, T, 2)
            range_xy_per_sample = max_xy_per_sample - min_xy_per_sample # shape (N, T, 2)
            invalid_range_mask = range_xy_per_sample < 0.01 # shape (N, T, 2)
            invalid_range_mask = invalid_range_mask.numpy().astype(bool) # shape (N, T, 2)
            invalid_range_mask = np.any(invalid_range_mask, axis=(1,2)) # shape (N,)
            all_motions_filtered = all_motions_filtered[~invalid_range_mask] # shape (N, T, 17, 3)
        else:
            invalid_range_mask = np.ones(N, dtype=bool)
            
        self.motions_2d = all_motions_filtered
        print(f"[JRDBDataset2D] : kept {len(self.motions_2d)} out of {total_tracks} tracks ({total_segments} segments, invalid range: {invalid_range_mask.sum()})")
                
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions_2d)

    def __getitem__(self, index):
        'Generates one sample of data'
        metadata = {
            "unique_track_identifier": f"JRDB_{index}",
            "recording": "JRDB",
            "flipped": False,
            "file_path": "unknown",
            "orig_dataset": "JRDB",
        }
        fake_label = torch.tensor(0)
        motion_2d = self.motions_2d[index]

        if self.flip and random.random()>0.5:                       
            motion_2d = flip_data(motion_2d)
            metadata["flipped"] = True

        motion_2d = torch.FloatTensor(motion_2d)
        return motion_2d, fake_label, metadata # motion_2d
    

class PosesInTheWildDataset2D(Dataset):
    def __init__(self, data_root, split, n_frames=48, data_stride=48, valid_threshold=0.25, min_valid_joints=9, subsample_factor=1, flip=True):
        super(PosesInTheWildDataset2D, self).__init__()
        self.n_frames = n_frames
        self.data_root = data_root
        self.flip = flip
        
        self.files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(os.path.join(self.data_root, split))) for f in fn if f.endswith(".pkl")]
        self.files.sort()
        
        self.motions_2d = []
        
        total_segments = 0
        total_subjects = 0
        for file in self.files:
            seq = pickle.load(open(file,'rb'), encoding='latin1')
            nsubjects = len(seq["genders"])
            for subject_id in range(nsubjects):
                total_subjects += 1
                subject_poses_2d = seq["poses2d"][subject_id] # N, 3, 18
                subject_poses_3d = seq["jointPositions"][subject_id] # N, 72
                subject_poses_2d = subject_poses_2d.swapaxes(1, 2) # N, 18, 3
                subject_poses_3d = subject_poses_3d.reshape(-1, 24, 3) # N, 24, 3
                
                subject_poses_2d = subject_poses_2d[::subsample_factor]
                subject_poses_3d = subject_poses_3d[::subsample_factor]
                
                subject_frames = subject_poses_2d.shape[0]
                for i in range(0, subject_frames, data_stride):
                    if i + n_frames > subject_frames:
                        continue
                    total_segments += 1
                    
                    subject_poses_2d_segment = subject_poses_2d[i:i+n_frames] # T, 18, 3
                    # subject_poses_3d_segment = subject_poses_3d[i:i+n_frames] # T, 24, 3
                    # check 2D validity
                    subject_poses_2d_segment[subject_poses_2d_segment[:,:,2] < valid_threshold] = 0
                    scores = subject_poses_2d_segment[:,:,2] # T, 18

                    valid_keypoints_mask = scores > 0 # T, 18
                    valid_keypoints_count = np.sum(valid_keypoints_mask, axis=1) # T
                    valid_keypoints_count_mask = valid_keypoints_count >= min_valid_joints # T
                    if np.all(valid_keypoints_count_mask):
                        self.motions_2d.append(subject_poses_2d_segment)
                    else:
                        continue
                    
                    # self.motions_2d.append(subject_poses_2d_segment)
                    # self.motions_3d.append(subject_poses_3d_segment)
        
        self.motions_2d = np.array(self.motions_2d) # N, T, 18, 3
        self.motions_2d = coco182h36m(self.motions_2d) # N, T, 17, 3
        
        print(f"[PosesInTheWildDataset2D] : kept {self.motions_2d.shape} out of {total_subjects} subjects ({total_segments} segments)")
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions_2d)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        metadata = {
            "unique_track_identifier": f"PosesInTheWild_{index}",
            "recording": "PosesInTheWild",
            "flipped": False,
            "file_path": "unknown",
            "orig_dataset": "PosesInTheWild",
        }
        fake_label = torch.tensor(0)
        motion_2d = self.motions_2d[index]
        
        if self.flip and random.random()>0.5:                       
            motion_2d = flip_data(motion_2d)
            metadata["flipped"] = True
            
        motion_2d = torch.FloatTensor(motion_2d)
        return motion_2d, fake_label, metadata # motion_2d
    
class Original_PoseTrackDataset2D(Dataset):
    def __init__(self, flip=True, scale_range=[0.25, 1]):
        super(Original_PoseTrackDataset2D, self).__init__()
        self.flip = flip
        data_root = "data/motion2d/posetrack18_annotations/train/"
        file_list = sorted(os.listdir(data_root))
        all_motions = []
        all_motions_filtered = []
        self.scale_range = scale_range
        for filename in file_list:
            with open(os.path.join(data_root, filename), 'r') as file:
                json_dict = json.load(file)
                annots = json_dict['annotations']
                imgs = json_dict['images']
                motions = defaultdict(list)
                for annot in annots:
                    tid = annot['track_id']
                    pose2d = np.array(annot['keypoints']).reshape(-1,3)
                    motions[tid].append(pose2d)
            all_motions += list(motions.values())
        for motion in all_motions:
            if len(motion)<30:
                continue
            motion = np.array(motion[:30])
            if np.sum(motion[:,:,2]) <= 306:  # Valid joint num threshold
                continue
            motion = crop_scale(motion, self.scale_range) 
            motion = posetrack2h36m(motion)
            motion[motion[:,:,2]==0] = 0
            if np.sum(motion[:,0,2]) < 30:
                continue                      # Root all visible (needed for framewise rootrel)
            all_motions_filtered.append(motion)
        all_motions_filtered = np.array(all_motions_filtered)
        self.motions_2d = all_motions_filtered
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions_2d)

    def __getitem__(self, index):
        'Generates one sample of data'
        motion_2d = torch.FloatTensor(self.motions_2d[index])
        if self.flip and random.random()>0.5:                       
            motion_2d = flip_data(motion_2d)
        return motion_2d, motion_2d
    
class Original_InstaVDataset2D(Dataset):
    def __init__(self, n_frames=81, data_stride=27, flip=True, valid_threshold=0.0, scale_range=[0.25, 1]):
        super(Original_InstaVDataset2D, self).__init__()
        self.flip = flip
        self.scale_range = scale_range
        motion_all = np.load('data/motion2d/InstaVariety/motion_all.npy')
        id_all = np.load('data/motion2d/InstaVariety/id_all.npy')
        split_id = split_clips(id_all, n_frames, data_stride)  
        motions_2d = motion_all[split_id]                        # [N, T, 17, 3]
        valid_idx = (motions_2d[:,0,0,2] > valid_threshold)
        self.motions_2d = motions_2d[valid_idx]
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions_2d)

    def __getitem__(self, index):
        'Generates one sample of data'
        motion_2d = self.motions_2d[index]
        motion_2d = crop_scale(motion_2d, self.scale_range) 
        motion_2d[motion_2d[:,:,2]==0] = 0
        if self.flip and random.random()>0.5:                       
            motion_2d = flip_data(motion_2d)
        motion_2d = torch.FloatTensor(motion_2d)
        return motion_2d, motion_2d
        