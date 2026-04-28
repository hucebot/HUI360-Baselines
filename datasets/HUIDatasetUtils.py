import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, ".."))

from mpmath import ellippi
import numpy as np
import pandas as pd
import copy
from math import e
import torch

from utils.print_utils import *
from utils.rle_tools import *
from utils.data_utils import *
from utils.sapiens_selected import SAPIENS_EXCLUDING_FACE_KEYPOINTS_NAMES

# Dataset filenames for offline mode as of 20/1/2026
LEGACY_DATASET_FILENAMES = ["data-0000-of-0068.csv", "data-0001-of-0068.csv", "data-0002-of-0068.csv", "data-0003-of-0068.csv", "data-0004-of-0068.csv", "data-0005-of-0068.csv", "data-0006-of-0068.csv", "data-0007-of-0068.csv", "data-0008-of-0068.csv", "data-0009-of-0068.csv", "data-0010-of-0068.csv", "data-0011-of-0068.csv", 
                          "data-0012-of-0068.csv", "data-0013-of-0068.csv", "data-0014-of-0068.csv", "data-0015-of-0068.csv", "data-0016-of-0068.csv", "data-0017-of-0068.csv", "data-0018-of-0068.csv", "data-0019-of-0068.csv", "data-0020-of-0068.csv", "data-0021-of-0068.csv", "data-0022-of-0068.csv", "data-0023-of-0068.csv", 
                          "data-0024-of-0068.csv", "data-0025-of-0068.csv", "data-0026-of-0068.csv", "data-0027-of-0068.csv", "data-0028-of-0068.csv", "data-0029-of-0068.csv", "data-0030-of-0068.csv", "data-0031-of-0068.csv", "data-0032-of-0068.csv", "data-0033-of-0068.csv", "data-0034-of-0068.csv", "data-0035-of-0068.csv", 
                          "data-0036-of-0068.csv", "data-0037-of-0068.csv", "data-0038-of-0068.csv", "data-0039-of-0068.csv", "data-0040-of-0068.csv", "data-0041-of-0068.csv", "data-0042-of-0068.csv", "data-0043-of-0068.csv", "data-0044-of-0068.csv", "data-0045-of-0068.csv", "data-0046-of-0068.csv", "data-0047-of-0068.csv", 
                          "data-0048-of-0068.csv", "data-0049-of-0068.csv", "data-0050-of-0068.csv", "data-0051-of-0068.csv", "data-0052-of-0068.csv", "data-0053-of-0068.csv", "data-0054-of-0068.csv", "data-0055-of-0068.csv", "data-0056-of-0068.csv", "data-0057-of-0068.csv", "data-0058-of-0068.csv", "data-0059-of-0068.csv", 
                          "data-0060-of-0068.csv", "data-0061-of-0068.csv", "data-0062-of-0068.csv", "data-0063-of-0068.csv", "data-0064-of-0068.csv", "data-0065-of-0068.csv", "data-0066-of-0068.csv", "data-0067-of-0068.csv", 
                          "ssupaug-0000-of-0028.csv", "ssupaug-0001-of-0028.csv", "ssupaug-0002-of-0028.csv", "ssupaug-0003-of-0028.csv", "ssupaug-0004-of-0028.csv", "ssupaug-0005-of-0028.csv", "ssupaug-0006-of-0028.csv", "ssupaug-0007-of-0028.csv", "ssupaug-0008-of-0028.csv", "ssupaug-0009-of-0028.csv", "ssupaug-0010-of-0028.csv", 
                          "ssupaug-0011-of-0028.csv", "ssupaug-0012-of-0028.csv", "ssupaug-0013-of-0028.csv", "ssupaug-0014-of-0028.csv", "ssupaug-0015-of-0028.csv", "ssupaug-0016-of-0028.csv", "ssupaug-0017-of-0028.csv", "ssupaug-0018-of-0028.csv", "ssupaug-0019-of-0028.csv", "ssupaug-0020-of-0028.csv", "ssupaug-0021-of-0028.csv", 
                          "ssupaug-0022-of-0028.csv", "ssupaug-0023-of-0028.csv", "ssupaug-0024-of-0028.csv", "ssupaug-0025-of-0028.csv", "ssupaug-0026-of-0028.csv", "ssupaug-0027-of-0028.csv"
                          ]

MAIN_DATASET_FILENAMES = [
    "data-2022_09_21_astor_place_landfill-0000-of-0097.csv", "data-2022_09_21_astor_place_recycle-0001-of-0097.csv", "data-2022_09_26_astor_place_landfill-0002-of-0097.csv", "data-2022_09_26_astor_place_recycle-0003-of-0097.csv", "data-2022_09_28_astor_place_landfill-0004-of-0097.csv", 
    "data-2022_09_28_astor_place_recycle-0005-of-0097.csv", "data-2022_10_06_astor_place_landfill-0006-of-0097.csv", "data-2022_10_06_astor_place_recycle-0007-of-0097.csv", "data-2022_10_12_astor_place_landfill_0-0008-of-0097.csv", "data-2022_10_12_astor_place_landfill_1-0009-of-0097.csv", 
    "data-2022_10_12_astor_place_recycle_0-0010-of-0097.csv", "data-2022_10_12_astor_place_recycle_1-0011-of-0097.csv", "data-2023_07_06_albee_square_landfill_0-0012-of-0097.csv", "data-2023_07_06_albee_square_landfill_1-0013-of-0097.csv", "data-2023_07_06_albee_square_recycle_0-0014-of-0097.csv", 
    "data-2023_07_06_albee_square_recycle_1-0015-of-0097.csv", "data-2023_07_07_albee_square_landfill_0-0016-of-0097.csv", "data-2023_07_07_albee_square_landfill_1-0017-of-0097.csv", "data-2023_07_07_albee_square_recycle_0-0018-of-0097.csv", "data-2023_07_07_albee_square_recycle_1-0019-of-0097.csv", 
    "data-2023_07_11_albee_square_landfill_1-0020-of-0097.csv", "data-2023_07_11_albee_square_recycle_0-0021-of-0097.csv", "data-2023_07_11_albee_square_recycle_1-0022-of-0097.csv", "data-2023_07_12_albee_square_landfill_0-0023-of-0097.csv", "data-2023_07_12_albee_square_recycle_0-0024-of-0097.csv", 
    "data-2023_07_12_albee_square_recycle_1-0025-of-0097.csv", "data-2023_07_14_albee_square_landfill-0026-of-0097.csv", "data-2023_07_14_albee_square_recycle-0027-of-0097.csv",
    "data-rosbag2_2025_07_07-10_24_20-0028-of-0097.csv","data-rosbag2_2025_07_07-10_49_31-0029-of-0097.csv","data-rosbag2_2025_07_07-11_16_10-0030-of-0097.csv","data-rosbag2_2025_07_07-12_38_45-0031-of-0097.csv","data-rosbag2_2025_07_07-15_33_32-0032-of-0097.csv","data-rosbag2_2025_07_10-15_47_18-0033-of-0097.csv",
    "data-rosbag2_2025_07_11-10_28_07-0034-of-0097.csv","data-rosbag2_2025_07_11-11_18_00-0035-of-0097.csv","data-rosbag2_2025_07_11-13_27_26-0036-of-0097.csv","data-rosbag2_2025_07_11-14_54_55-0037-of-0097.csv","data-rosbag2_2025_07_15-10_29_43-0038-of-0097.csv","data-rosbag2_2025_07_15-12_39_21-0039-of-0097.csv",
    "data-rosbag2_2025_07_15-13_41_01-0040-of-0097.csv","data-rosbag2_2025_07_15-14_48_22-0041-of-0097.csv","data-rosbag2_2025_07_16-13_12_03-0042-of-0097.csv","data-rosbag2_2025_07_16-14_07_49-0043-of-0097.csv","data-rosbag2_2025_07_16-15_50_45-0044-of-0097.csv","data-rosbag2_2025_07_17-11_28_34-0045-of-0097.csv",
    "data-rosbag2_2025_07_17-12_52_12-0046-of-0097.csv","data-rosbag2_2025_07_18-10_37_07-0047-of-0097.csv","data-rosbag2_2025_07_21-10_22_11-0048-of-0097.csv","data-rosbag2_2025_07_21-11_56_40-0049-of-0097.csv","data-rosbag2_2025_07_21-13_09_22-0050-of-0097.csv","data-rosbag2_2025_07_21-14_11_37-0051-of-0097.csv",
    "data-rosbag2_2025_07_21-15_15_07-0052-of-0097.csv","data-rosbag2_2025_07_22-09_38_18-0053-of-0097.csv","data-rosbag2_2025_07_22-10_59_25-0054-of-0097.csv","data-rosbag2_2025_07_22-12_18_30-0055-of-0097.csv","data-rosbag2_2025_07_22-13_30_39-0056-of-0097.csv","data-rosbag2_2025_07_23-11_01_56-0057-of-0097.csv",
    "data-rosbag2_2025_07_23-12_18_45-0058-of-0097.csv","data-rosbag2_2025_07_23-13_17_40-0059-of-0097.csv","data-rosbag2_2025_07_23-14_30_55-0060-of-0097.csv","data-rosbag2_2025_07_24-10_41_01-0061-of-0097.csv","data-rosbag2_2025_07_24-12_14_56-0062-of-0097.csv","data-rosbag2_2025_07_24-13_33_54-0063-of-0097.csv",
    "data-rosbag2_2025_07_24-14_33_36-0064-of-0097.csv","data-rosbag2_2025_07_25-10_52_32-0065-of-0097.csv","data-rosbag2_2025_07_25-14_14_16-0066-of-0097.csv","data-rosbag2_2025_07_28-10_18_10-0067-of-0097.csv","data-rosbag2_2025_07_28-11_25_33-0068-of-0097.csv","data-rosbag2_2025_07_28-13_05_46-0069-of-0097.csv",
    "data-rosbag2_2025_07_28-14_19_07-0070-of-0097.csv","data-rosbag2_2025_07_29-10_23_14-0071-of-0097.csv","data-rosbag2_2025_07_29-13_17_18-0072-of-0097.csv","data-rosbag2_2025_07_29-14_09_46-0073-of-0097.csv","data-rosbag2_2025_10_07-15_03_48-0074-of-0097.csv","data-rosbag2_2025_10_07-16_21_39-0075-of-0097.csv",
    "data-rosbag2_2025_10_09-08_52_20-0076-of-0097.csv","data-rosbag2_2025_10_09-10_23_38-0077-of-0097.csv","data-rosbag2_2025_10_09-17_37_23-0078-of-0097.csv","data-rosbag2_2025_10_09-18_50_21-0079-of-0097.csv","data-rosbag2_2025_10_15-09_37_49-0080-of-0097.csv","data-rosbag2_2025_10_15-11_03_05-0081-of-0097.csv",
    "data-rosbag2_2025_10_15-12_02_00-0082-of-0097.csv","data-rosbag2_2025_10_15-12_27_14-0083-of-0097.csv","data-rosbag2_2025_10_15-13_11_27-0084-of-0097.csv","data-rosbag2_2025_10_15-14_02_29-0085-of-0097.csv","data-rosbag2_2025_10_15-14_30_03-0086-of-0097.csv","data-rosbag2_2025_10_16-09_27_48-0087-of-0097.csv",
    "data-rosbag2_2025_10_16-11_29_56-0088-of-0097.csv","data-rosbag2_2025_10_16-12_47_57-0089-of-0097.csv","data-rosbag2_2025_10_17-13_19_29-0090-of-0097.csv","data-rosbag2_2025_10_17-14_28_15-0091-of-0097.csv","data-rosbag2_2025_10_17-15_11_00-0092-of-0097.csv","data-rosbag2_2025_10_17-16_47_09-0093-of-0097.csv",
    "data-rosbag2_2025_10_20-10_13_32-0094-of-0097.csv","data-rosbag2_2025_10_20-11_51_02-0095-of-0097.csv","data-rosbag2_2025_10_20-16_02_46-0096-of-0097.csv"
    ]


def convert_to_polar_coordinates(input_tensor: torch.Tensor, data_columns: list) -> tuple[torch.Tensor, list]:
    """ Convert the input tensor to polar coordinates
    
    Args:
        input_tensor (torch.Tensor): input tensor of shape (B, T, D)
        data_columns (list): data columns (list of strings) used to find the index in the original tensor of the x, y and score columns for each joint
        
    Returns:
        torch.Tensor: output tensor of shape (B, T, D)
    """
    
    B,T,D = input_tensor.shape
    assert(len(data_columns) == D), f"Data columns length mismatch: {len(data_columns)} != {D}"
    
    output_dim = len(data_columns)
    for col in data_columns:
        if col.endswith("_x") or col.endswith("_y"):
            output_dim += 1
        if col == "xmin" or col == "xmax" or col == "ymin" or col == "ymax":
            output_dim += 1
            
    output_tensor = torch.zeros((B,T,output_dim), device=input_tensor.device)
    output_data_columns = []
    
    new_idx = 0
    for idx, col in enumerate(data_columns):
        if col.endswith("_x") or col == "xmin" or col == "xmax":
            # convert 0-1 to -pi to pi
            output_tensor[:, :, new_idx] = torch.sin(torch.pi * (2 * (input_tensor[:, :, idx] - 0.5)))
            new_idx += 1
            output_tensor[:, :, new_idx] = torch.cos(torch.pi * (2 * (input_tensor[:, :, idx] - 0.5)))
            new_idx += 1
            output_data_columns.append(col + "_sin")
            output_data_columns.append(col + "_cos")
            
        elif col.endswith("_y") or col == "ymin" or col == "ymax":
            # convert 0-1 to -pi/2 to pi/2
            output_tensor[:, :, new_idx] = torch.sin((torch.pi/2) * (2 * (input_tensor[:, :, idx] - 0.5)))
            new_idx += 1
            output_tensor[:, :, new_idx] = torch.cos((torch.pi/2) * (2 * (input_tensor[:, :, idx] - 0.5)))
            new_idx += 1
            output_data_columns.append(col + "_sin")
            output_data_columns.append(col + "_cos")
            
        else:
            output_tensor[:, :, new_idx] = input_tensor[:, :, idx]
            output_data_columns.append(col)

    return output_tensor, output_data_columns
    
        
def input_tensor_to_format_by_channel(input_tensor, metadata, data_columns, remove_joints: list = None):
    """ Rearange the input tensor by channels (_x, _y, _score)
    
    Important we assume V = 17 and C = 3 for now (ViTPose/COCO17 keypoints and channels)
    
    Args:
        input_tensor (torch.Tensor): input tensor of shape (B, T, D)
        metadata (dict): metadata (unused for now)
        data_columns (list): data columns (list of strings) used to find the index in the original tensor of the x, y and score columns for each joint
        
    Returns:
        torch.Tensor: output tensor of shape (B, T, V, C)
    """
    
    # print("metadata : ", metadata)
    # print("data_columns : ", data_columns)
    
    B,T,D = input_tensor.shape
    assert(len(data_columns) == D), f"Data columns length mismatch: {len(data_columns)} != {D}"
    
    vitpose_joints = VITPOSE_KEYPOINTS_NAMES
    
    C = 3
    V = 17
    output_tensor = torch.zeros((B,T,V,C), device=input_tensor.device)
    
    for joint_idx, joint in enumerate(vitpose_joints):
        assert(f"vitpose_{joint}_x" in data_columns), f"vitpose_{joint}_x not in data_columns"
        assert(f"vitpose_{joint}_y" in data_columns), f"vitpose_{joint}_y not in data_columns"
        assert(f"vitpose_{joint}_score" in data_columns), f"vitpose_{joint}_score not in data_columns"
        x_idx = data_columns.index(f"vitpose_{joint}_x")
        y_idx = data_columns.index(f"vitpose_{joint}_y")
        score_idx = data_columns.index(f"vitpose_{joint}_score")
        
        output_tensor[:, :, joint_idx, 0] = input_tensor[:, :, x_idx]
        output_tensor[:, :, joint_idx, 1] = input_tensor[:, :, y_idx]
        output_tensor[:, :, joint_idx, 2] = input_tensor[:, :, score_idx]
        
    if remove_joints is not None:
        for joint in remove_joints:
            output_tensor[:, :, joint, :2] = -1.0
            output_tensor[:, :, joint, 2] = 0.0
        
    return output_tensor


def input_tensor_to_format_by_channel_sapiens_without_face(input_tensor, metadata, data_columns):
    """ Rearange the input tensor by channels (_x, _y, _score)
    
    Important we assume V = 63 and C = 3 for now (Sapiens without face keypoints and channels)
    
    Args:
        input_tensor (torch.Tensor): input tensor of shape (B, T, D)
        metadata (dict): metadata (unused for now)
        data_columns (list): data columns (list of strings) used to find the index in the original tensor of the x, y and score columns for each joint
        
    Returns:
        torch.Tensor: output tensor of shape (B, T, V, C) with V = 63
    """
    
    # print("metadata : ", metadata)
    # print("data_columns : ", data_columns)
    
    B,T,D = input_tensor.shape
    assert(len(data_columns) == D), f"Data columns length mismatch: {len(data_columns)} != {D}"
    
    sapiens_joints = SAPIENS_EXCLUDING_FACE_KEYPOINTS_NAMES
    
    C = 3
    V = 63
    output_tensor = torch.zeros((B,T,V,C), device=input_tensor.device)
    
    for joint_idx, joint in enumerate(sapiens_joints):
        assert(f"{joint}_x" in data_columns), f"sapiens_{joint}_x not in data_columns"
        assert(f"{joint}_y" in data_columns), f"sapiens_{joint}_y not in data_columns"
        assert(f"{joint}_score" in data_columns), f"sapiens_{joint}_score not in data_columns"
        x_idx = data_columns.index(f"{joint}_x")
        y_idx = data_columns.index(f"{joint}_y")
        score_idx = data_columns.index(f"{joint}_score")
        
        output_tensor[:, :, joint_idx, 0] = input_tensor[:, :, x_idx]
        output_tensor[:, :, joint_idx, 1] = input_tensor[:, :, y_idx]
        output_tensor[:, :, joint_idx, 2] = input_tensor[:, :, score_idx]
        
    return output_tensor
    
    
def crop_scale_torch(motion, scale_range=[1, 1]):
    '''
        Args:
            motion: torch.tensor of shape (..., T, 17, 3)
            scale_range: list or tuple of two floats, min and max scale
        Returns:
            torch.tensor of same shape as motion, normalized to [-1, 1] (on x/y channels)
    '''

    result = motion.clone()
    # valid where score channel != 0
    valid_mask = (motion[..., 2] != 0)
    valid_coords = motion[..., :2][valid_mask]
    if valid_coords.shape[0] < 4:
        return torch.zeros_like(motion)
    xmin = valid_coords[:, 0].min()
    xmax = valid_coords[:, 0].max()
    ymin = valid_coords[:, 1].min()
    ymax = valid_coords[:, 1].max()
    
    if scale_range[0] == scale_range[1]:
        ratio = scale_range[0] # typically 1
    else:
        # PyTorch doesn't have a direct np.random.uniform, use torch version
        ratio = torch.empty(1).uniform_(float(scale_range[0]), float(scale_range[1]))[0]
        
    scale = max(xmax - xmin, ymax - ymin) * ratio
    if scale == 0:
        return torch.zeros_like(motion)
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    # Crop and scale
    result[..., :2] = (motion[..., :2] - torch.tensor([xs, ys], dtype=motion.dtype, device=motion.device)) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = torch.clamp(result, -1, 1)
    return result

def crop_scale_torch_by_sample(motion):
    '''
        Args:
            motion: torch.tensor of shape (B, T, 17, 3)
        Returns:
            torch.tensor of shape (B, T, 17, 3), normalized to [-1, 1] (on x/y channels)
    '''

    assert(motion.ndim == 4), "Motion must be of shape (B, T, 17, 3)"

    result = motion.clone()
    B, T, V, C = motion.shape
    
    # valid where score channel > 0, shape: (B, T, 17)
    # valid_mask = (motion[..., 2] != 0)
    valid_mask = (motion[..., 2] > 0)
    
    # Count valid coordinates per sample, shape: (B,)
    num_valid_per_sample = valid_mask.sum(dim=(1, 2))
    
    # Find samples with insufficient valid coordinates (< 4)
    invalid_samples_mask = num_valid_per_sample < 4 # shape: (B,)
    
    # Reshape for easier processing: (B, T*V, 3) -> (B, T*V, 2) for coordinates
    coords = motion[..., :2].reshape(B, T * V, 2)
    valid_mask_flat = valid_mask.reshape(B, T * V)
    
    # Set invalid coordinates to a large value for min/max computation
    # Use inf/-inf so they don't affect min/max calculations
    coords_for_min = coords.clone()
    coords_for_max = coords.clone()
    coords_for_min[~valid_mask_flat] = float('inf')
    coords_for_max[~valid_mask_flat] = float('-inf')
    
    # Compute min/max per sample, shape: (B, 2) where [:, 0] is x, [:, 1] is y
    xmin_per_sample = coords_for_min[:, :, 0].min(dim=1)[0]  # (B,)
    xmax_per_sample = coords_for_max[:, :, 0].max(dim=1)[0]  # (B,)
    ymin_per_sample = coords_for_min[:, :, 1].min(dim=1)[0]  # (B,)
    ymax_per_sample = coords_for_max[:, :, 1].max(dim=1)[0]  # (B,)
    
    # Compute scale per sample, shape: (B,)
    x_range = xmax_per_sample - xmin_per_sample
    y_range = ymax_per_sample - ymin_per_sample
    scale_per_sample = torch.maximum(x_range, y_range) # shape: (B,)
    
    # Find samples with zero scale
    zero_scale_mask = scale_per_sample == 0
    
    # Combine invalid masks
    samples_to_zero = invalid_samples_mask | zero_scale_mask # shape: (B,)
    
    # Compute crop centers per sample, shape: (B, 2)
    xs_per_sample = (xmin_per_sample + xmax_per_sample - scale_per_sample) / 2
    ys_per_sample = (ymin_per_sample + ymax_per_sample - scale_per_sample) / 2
    
    # Set scale to 1 for invalid samples to avoid division by zero (will be zeroed out later)
    scale_per_sample_safe = torch.where(scale_per_sample > 0, scale_per_sample, torch.ones_like(scale_per_sample))
    
    # Expand for broadcasting: (B, 1, 1, 2) to match (B, T, 17, 2)
    crop_centers = torch.stack([xs_per_sample, ys_per_sample], dim=-1).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, 2)
    scale_per_sample_expanded = scale_per_sample_safe.view(B, 1, 1, 1)  # (B, 1, 1, 1)
    
    # Crop and scale in vectorized way
    result[..., :2] = (motion[..., :2] - crop_centers) / scale_per_sample_expanded
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = torch.clamp(result, -1, 1)
    
    # Zero out invalid samples
    result[samples_to_zero] = 0.0
    
    return result

def crop_scale(motion, scale_range=[1, 1]):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]!=0][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result

def coco2nwucla(x):
    """ Convert COCO format to NW-UCLA format

    COCO:
        1-nose
        2-left_eye
        3-right_eye
        4-left_ear
        5-right_ear
        6-left_shoulder
        7-right_shoulder
        8-left_elbow
        9-right_elbow
        10-left_wrist
        11-right_wrist
        12-left_hip
        13-right_hip
        14-left_knee
        15-right_knee
        16-left_ankle
        17-right_ankle
        
    NW-UCLA:
        0-base spine
        1-middle spine
        2-neck
        3-head
        4-left shoulder
        5-left elbow
        6-left wrist
        7-left hand
        8-right shoulder
        9-right elbow
        10-right wrist
        11-right hand
        12-left hip
        13-left knee
        14-left ankle
        15-left foot
        16-right hip
        17-right knee
        18-right ankle
        19-right foot
        
        
    Args:
        x (torch.Tensor): input tensor of shape (B, T, 17, C)
        
    Returns:
        torch.Tensor: output tensor of shape (B, T, 20, 3)
    """
    
    B, T, _, C = x.shape
    device = x.device

    nwucla = torch.zeros((B, T, 20, C), device=device)

    # COCO joints
    nose = x[:, :,  0, :]
    l_sh = x[:, :,  5, :]
    r_sh = x[:, :,  6, :]
    l_el = x[:, :,  7, :]
    r_el = x[:, :,  8, :]
    l_wr = x[:, :,  9, :]
    r_wr = x[:, :, 10, :]
    l_hp = x[:, :, 11, :]
    r_hp = x[:, :, 12, :]
    l_kn = x[:, :, 13, :]
    r_kn = x[:, :, 14, :]
    l_an = x[:, :, 15, :]
    r_an = x[:, :, 16, :]

    # Spine approximations
    hip_center = (l_hp + r_hp) / 2
    shoulder_center = (l_sh + r_sh) / 2 # also spine
    neck = (nose + shoulder_center) / 2
    spine_mid = (hip_center + shoulder_center) / 2

    # Spine approximations
    hip_center = (l_hp + r_hp) / 2
    shoulder_center = (l_sh + r_sh) / 2
    spine_mid = (hip_center + shoulder_center) / 2

    # NW-UCLA mapping
    nwucla[:, :, 0]  = hip_center       # base spine
    nwucla[:, :, 1]  = spine_mid        # middle spine
    nwucla[:, :, 2]  = shoulder_center  # neck
    nwucla[:, :, 3]  = nose             # head

    nwucla[:, :, 4]  = l_sh             # left shoulder
    nwucla[:, :, 5]  = l_el             # left elbow
    nwucla[:, :, 6]  = l_wr             # left wrist
    nwucla[:, :, 7]  = l_wr             # left hand

    nwucla[:, :, 8]  = r_sh             # right shoulder
    nwucla[:, :, 9]  = r_el             # right elbow
    nwucla[:, :,10]  = r_wr             # right wrist
    nwucla[:, :,11]  = r_wr             # right hand

    nwucla[:, :,12]  = l_hp             # left hip
    nwucla[:, :,13]  = l_kn             # left knee
    nwucla[:, :,14]  = l_an             # left ankle
    nwucla[:, :,15]  = l_an             # left foot

    nwucla[:, :,16]  = r_hp             # right hip
    nwucla[:, :,17]  = r_kn             # right knee
    nwucla[:, :,18]  = r_an             # right ankle
    nwucla[:, :,19]  = r_an             # right foot

    return nwucla

def coco2nturgbd(x):
    """ Convert COCO format to NTURGBD format

    COCO:
        1-nose
        2-left_eye
        3-right_eye
        4-left_ear
        5-right_ear
        6-left_shoulder
        7-right_shoulder
        8-left_elbow
        9-right_elbow
        10-left_wrist
        11-right_wrist
        12-left_hip
        13-right_hip
        14-left_knee
        15-right_knee
        16-left_ankle
        17-right_ankle
        
    NTURGBD:
        1—base of the spine
        2—middle of the spine
        3—neck
        4—head
        5—left shoulder
        6—left elbow
        7—left wrist
        8—left hand
        9—right shoulder
        10—right elbow
        11—right wrist
        12—right hand
        13—left hip
        14—left knee
        15—left ankle
        16—left foot
        17—right hip
        18—right knee
        19—right ankle
        20—right foot
        21—spine
        22—tip of the left hand
        23—left thumb
        24—tip of the right hand
        25—right thumb.
        
        
    Args:
        x (torch.Tensor): input tensor of shape (B, T, 17, C)
        
    Returns:
        torch.Tensor: output tensor of shape (B, T, 25, 3)
    """
    
    B, T, _, C = x.shape
    device = x.device

    ntu = torch.zeros((B, T, 25, C), device=device)

    # COCO joints
    nose = x[:, :,  0, :]
    l_sh = x[:, :,  5, :]
    r_sh = x[:, :,  6, :]
    l_el = x[:, :,  7, :]
    r_el = x[:, :,  8, :]
    l_wr = x[:, :,  9, :]
    r_wr = x[:, :, 10, :]
    l_hp = x[:, :, 11, :]
    r_hp = x[:, :, 12, :]
    l_kn = x[:, :, 13, :]
    r_kn = x[:, :, 14, :]
    l_an = x[:, :, 15, :]
    r_an = x[:, :, 16, :]

    # Spine approximations
    hip_center = (l_hp + r_hp) / 2
    shoulder_center = (l_sh + r_sh) / 2 # also spine
    neck = (nose + shoulder_center) / 2
    spine_mid = (hip_center + shoulder_center) / 2


    # NTU mapping
    ntu[:, :, 0, :]  = hip_center              # base spine
    ntu[:, :, 1, :]  = spine_mid               # middle spine
    ntu[:, :, 2, :]  = neck                    # neck
    ntu[:, :, 3, :]  = nose                    # head

    ntu[:, :, 4, :]  = l_sh                    # left shoulder
    ntu[:, :, 5, :]  = l_el                    # left elbow
    ntu[:, :, 6, :]  = l_wr                    # left wrist
    ntu[:, :, 7, :]  = l_wr                    # left hand

    ntu[:, :, 8, :]  = r_sh                    # right shoulder
    ntu[:, :, 9, :]  = r_el                    # right elbow
    ntu[:, :,10, :]  = r_wr                    # right wrist
    ntu[:, :,11, :]  = r_wr                    # right hand

    ntu[:, :,12, :]  = l_hp                    # left hip
    ntu[:, :,13, :]  = l_kn                    # left knee
    ntu[:, :,14, :]  = l_an                    # left ankle
    ntu[:, :,15, :]  = l_an                    # left foot

    ntu[:, :,16, :]  = r_hp                    # right hip
    ntu[:, :,17, :]  = r_kn                    # right knee
    ntu[:, :,18, :]  = r_an                    # right ankle
    ntu[:, :,19, :]  = r_an                    # right foot

    ntu[:, :,20, :]  = shoulder_center         # spine
    ntu[:, :,21, :]  = l_wr                    # left hand tip
    ntu[:, :,22, :]  = l_wr                    # left thumb
    ntu[:, :,23, :]  = r_wr                    # right hand tip
    ntu[:, :,24, :]  = r_wr                    # right thumb

    return ntu





def coco2nturgbd_nospine_mid(x):
    """ Convert COCO format to NTURGBD format

    COCO:
        1-nose
        2-left_eye
        3-right_eye
        4-left_ear
        5-right_ear
        6-left_shoulder
        7-right_shoulder
        8-left_elbow
        9-right_elbow
        10-left_wrist
        11-right_wrist
        12-left_hip
        13-right_hip
        14-left_knee
        15-right_knee
        16-left_ankle
        17-right_ankle
        
    NTURGBD:
        1—base of the spine
        # 2—middle of the spine
        2—neck
        3—head
        4—left shoulder
        5—left elbow
        6—left wrist
        7—left hand
        8—right shoulder
        9—right elbow
        10—right wrist
        11—right hand
        12—left hip
        13—left knee
        14—left ankle
        15—left foot
        16—right hip
        17—right knee
        18—right ankle
        19—right foot
        20—spine
        21—tip of the left hand
        22—left thumb
        23—tip of the right hand
        24—right thumb.
        
        
    Args:
        x (torch.Tensor): input tensor of shape (B, T, 17, C)
        
    Returns:
        torch.Tensor: output tensor of shape (B, T, 24, 3)
    """
    
    B, T, _, C = x.shape
    device = x.device

    ntu = torch.zeros((B, T, 24, C), device=device)

    # COCO joints
    nose = x[:, :,  0, :]
    l_sh = x[:, :,  5, :]
    r_sh = x[:, :,  6, :]
    l_el = x[:, :,  7, :]
    r_el = x[:, :,  8, :]
    l_wr = x[:, :,  9, :]
    r_wr = x[:, :, 10, :]
    l_hp = x[:, :, 11, :]
    r_hp = x[:, :, 12, :]
    l_kn = x[:, :, 13, :]
    r_kn = x[:, :, 14, :]
    l_an = x[:, :, 15, :]
    r_an = x[:, :, 16, :]

    # Spine approximations
    hip_center = (l_hp + r_hp) / 2
    shoulder_center = (l_sh + r_sh) / 2 # also spine
    neck = (nose + shoulder_center) / 2
    spine_mid = (hip_center + shoulder_center) / 2

    # NTU mapping
    ntu[:, :, 0, :]  = hip_center              # base spine
    # ntu[:, :, 1, :]  = spine_mid               # middle spine
    ntu[:, :, 1, :]  = neck         # neck
    ntu[:, :, 2, :]  = nose                    # head

    ntu[:, :, 3, :]  = l_sh                    # left shoulder
    ntu[:, :, 4, :]  = l_el                    # left elbow
    ntu[:, :, 5, :]  = l_wr                    # left wrist
    ntu[:, :, 6, :]  = l_wr                    # left hand

    ntu[:, :, 7, :]  = r_sh                    # right shoulder
    ntu[:, :, 8, :]  = r_el                    # right elbow
    ntu[:, :, 9, :]  = r_wr                    # right wrist
    ntu[:, :,10, :]  = r_wr                    # right hand

    ntu[:, :,11, :]  = l_hp                    # left hip
    ntu[:, :,12, :]  = l_kn                    # left knee
    ntu[:, :,13, :]  = l_an                    # left ankle
    ntu[:, :,14, :]  = l_an                    # left foot

    ntu[:, :,15, :]  = r_hp                    # right hip
    ntu[:, :,16, :]  = r_kn                    # right knee
    ntu[:, :,17, :]  = r_an                    # right ankle
    ntu[:, :,18, :]  = r_an                    # right foot

    ntu[:, :,19, :]  = shoulder_center         # spine
    ntu[:, :,20, :]  = l_wr                    # left hand tip
    ntu[:, :,21, :]  = l_wr                    # left thumb
    ntu[:, :,22, :]  = r_wr                    # right hand tip
    ntu[:, :,23, :]  = r_wr                    # right thumb

    return ntu


def sapiensnoface2nturgbd_nospine_mid(x):
    """ Convert COCO format to NTURGBD format

    SAPIENS NO FACE (63 keypoints):
        1-nose
        2-left_eye
        3-right_eye
        4-left_ear
        5-right_ear
        6-left_shoulder
        7-right_shoulder
        8-left_elbow
        9-right_elbow
        10-left_hip
        11-right_hip
        12-left_knee
        13-right_knee
        14-left_ankle
        15-right_ankle
        ...
        16-left_big_toe
        ...
        19-right_big_toe
        ...
        22-right_thumb4
        ...
        30-right_middle_finger4
        ...
        42-right_wrist
        ...
        43-left_thumb4
        ...
        51-left_middle_finger4
        ...
        63-left_wrist
        
    NTURGBD:
        1—base of the spine
        # 2—middle of the spine
        2—neck
        3—head
        4—left shoulder
        5—left elbow
        6—left wrist
        7—left hand
        8—right shoulder
        9—right elbow
        10—right wrist
        11—right hand
        12—left hip
        13—left knee
        14—left ankle
        15—left foot
        16—right hip
        17—right knee
        18—right ankle
        19—right foot
        20—spine
        21—tip of the left hand
        22—left thumb
        23—tip of the right hand
        24—right thumb.
        
        
    Args:
        x (torch.Tensor): input tensor of shape (B, T, 17, C)
        
    Returns:
        torch.Tensor: output tensor of shape (B, T, 24, 3)
    """
    
    B, T, _, C = x.shape
    device = x.device

    ntu = torch.zeros((B, T, 24, C), device=device)
    
    # Sapien no face joints
    nose = x[:, :,  0, :]
    l_sh = x[:, :,  5, :]
    r_sh = x[:, :,  6, :]
    l_el = x[:, :,  7, :]
    r_el = x[:, :,  8, :]
    l_wr = x[:, :, 62, :]
    r_wr = x[:, :, 41, :]
    l_hp = x[:, :, 9, :]
    r_hp = x[:, :, 10, :]
    l_kn = x[:, :, 11, :]
    r_kn = x[:, :, 12, :]
    l_an = x[:, :, 13, :]
    r_an = x[:, :, 14, :]
    l_bt = x[:, :, 15, :]
    r_bt = x[:, :, 18, :]
    
    r_th = x[:, :, 21, :]
    r_mf = x[:, :, 29, :]
    
    l_th = x[:, :, 42, :]
    l_mf = x[:, :, 50, :]

    # Spine approximations
    hip_center = (l_hp + r_hp) / 2
    shoulder_center = (l_sh + r_sh) / 2 # also spine
    neck = (nose + shoulder_center) / 2
    spine_mid = (hip_center + shoulder_center) / 2

    left_hand = (l_wr + l_mf) / 2 # between wrist and mf
    right_hand = (r_wr + r_mf) / 2 # between wrist and mf

    # NTU mapping
    ntu[:, :, 0, :]  = hip_center              # base spine
    # ntu[:, :, 1, :]  = spine_mid               # middle spine
    ntu[:, :, 1, :]  = neck         # neck
    ntu[:, :, 2, :]  = nose                    # head

    ntu[:, :, 3, :]  = l_sh                    # left shoulder
    ntu[:, :, 4, :]  = l_el                    # left elbow
    ntu[:, :, 5, :]  = l_wr                    # left wrist
    ntu[:, :, 6, :]  = left_hand               # left hand

    ntu[:, :, 7, :]  = r_sh                    # right shoulder
    ntu[:, :, 8, :]  = r_el                    # right elbow
    ntu[:, :, 9, :]  = r_wr                    # right wrist
    ntu[:, :,10, :]  = right_hand              # right hand

    ntu[:, :,11, :]  = l_hp                    # left hip
    ntu[:, :,12, :]  = l_kn                    # left knee
    ntu[:, :,13, :]  = l_an                    # left ankle
    ntu[:, :,14, :]  = l_bt                    # left foot (use big toe)

    ntu[:, :,15, :]  = r_hp                    # right hip
    ntu[:, :,16, :]  = r_kn                    # right knee
    ntu[:, :,17, :]  = r_an                    # right ankle
    ntu[:, :,18, :]  = r_bt                    # right foot (use big toe)

    ntu[:, :,19, :]  = shoulder_center         # spine
    ntu[:, :,20, :]  = l_mf                    # left hand tip (use middle finger)
    ntu[:, :,21, :]  = l_th                    # left thumb
    ntu[:, :,22, :]  = r_mf                    # right hand tip (use middle finger)
    ntu[:, :,23, :]  = r_th                    # right thumb

    return ntu

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
    y = torch.zeros(x.shape, device=x.device)
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
    

def keypoints17_to_coco18(kps):
    """
    Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
    New keypoint (neck) is the average of the shoulders, and points
    are also reordered.
    """
    kp_np = np.array(kps)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=np.uint8)
    kp_coco18 = kp_np[..., opp_order, :]
    return kp_coco18

def keypoints17_to_coco18_torch(kps):
    """
    Convert a 17 keypoints coco format skeleton to an 18 keypoint one using torch operations.
    Preserves tensor data type and device.
    New keypoint (neck) is the average of the shoulders, and points are also reordered.
    Args:
        kps (torch.Tensor): Keypoints tensor of shape (..., 17, C)
    Returns:
        torch.Tensor: Reordered tensor of shape (..., 18, C)
    """
    # Compute neck keypoint
    neck_kp_vec = 0.5 * (kps[..., 5, :] + kps[..., 6, :])
    neck_kp_vec = neck_kp_vec.unsqueeze(-2)
    # Concatenate neck
    kp_with_neck = torch.cat([kps, neck_kp_vec], dim=-2)
    # Order for COCO18 (following original code)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = torch.tensor(opp_order, dtype=torch.long, device=kp_with_neck.device)
    kp_coco18 = kp_with_neck.index_select(-2, opp_order)
    return kp_coco18

def get_contiguous_true_segments(array: np.array) -> list:
    """ Get the contiguous true segments of a boolean array
    
    Example : [False,False,True,True,False,True,True,True] returns [(2, 2), (5, 3)]
    
    Args:
        array (np.array): boolean array
        
    Returns:
        list: list of tuples (start, length) with the positions of the true segments and their lengths. Inclusive of the first element.
    """
    
    assert(array.dtype == bool), "array must be a boolean array"
    assert(array.ndim == 1), "array must be a 1D array"
    assert(np.sum(array) > 0), "array must have at least one True"
    #Indices are found here
    segment_begin_positions = np.argwhere(np.diff(array,prepend=False,append=False)) # whenever it shifts, we add Falses, at both ends, since we are searching for Trues
    assert(len(segment_begin_positions) % 2 == 0), "array should always yield an even number of segment begin positions"
    #Conversion into list of 2-tuples
    segments_from_to = segment_begin_positions.reshape(len(segment_begin_positions)//2,2)
    segments_from_to = [tuple(r) for r in segments_from_to] # start position to start of the next segment (ie a segment of Falses)
    segments_from_length = [(start, to - start) for (start, to) in segments_from_to]
    return segments_from_length


def ssupaug_dataset_handling(recording_dataset: pd.DataFrame, 
                             recording_name: str, 
                             verbose: bool = True, 
                             add_sapiens_columns: bool = True,
                             filter_boxes: bool = True) -> pd.DataFrame:
    
    """ Special function to handle the ssupaug dataset for two possible issues : missing sapiens columns (not run yet), and recentering of boxes (some were outside of the frame)
    Also allow to perform track selection in advance to save memory
    """
    columns_in_recording_dataset = recording_dataset.columns

    if filter_boxes:
        # first drop rows if xmin is bigger than xmax (it should not happen but just in case)
        xmin_bigger_than_xmax_mask = recording_dataset["xmin"] > recording_dataset["xmax"]
        recording_dataset = recording_dataset[~xmin_bigger_than_xmax_mask]
        # prDebug(f"[{recording_name}] Dropping {xmin_bigger_than_xmax_mask.sum()/len(recording_dataset)*100:.2f}% rows where xmin is bigger than xmax")
        
        # then drop rows if width is more than half of the image width
        bbox_width_mask = recording_dataset["xmax"] - recording_dataset["xmin"] > (recording_dataset["image_width"]/2)
        recording_dataset = recording_dataset[~bbox_width_mask]
        # prDebug(f"[{recording_name}] Dropping {bbox_width_mask.sum()/len(recording_dataset)*100:.2f}% rows where width is more than half of the image width")
        
    if add_sapiens_columns:
        # check that there is 0 column starting with "sapiens"
        ncolumns_sapiens = len([col for col in columns_in_recording_dataset if col.startswith("sapiens")])
        # prDebug(f"[{recording_name}] Number of sapiens columns: {ncolumns_sapiens}")
        if ncolumns_sapiens == 308*3:
            # this is expected
            pass
        elif ncolumns_sapiens == 0:
            prWarning(f"[{recording_name}] There is no sapiens columns, padding with 0s")
            #there is no sapiens columns, add fake ones
            sapiens_columns = [col for col in FULL_DATA_COLUMNS if col.startswith("sapiens")]
            fake_sapiens_cols = pd.DataFrame(np.zeros((len(recording_dataset), len(sapiens_columns))), columns=sapiens_columns, index=recording_dataset.index)
            recording_dataset = pd.concat([recording_dataset, fake_sapiens_cols], axis=1)
        else:
            raise ValueError(f"Unexpected number of sapiens columns: {len(ncolumns_sapiens)} (expected 308*3 or 0)")
    
    return recording_dataset
                

def process_csv(args):
    csv, hf_data_dir, include_recordings, include_columns, return_masks, verbose = args
    original_include_columns = copy.deepcopy(include_columns) # copy to avoid modifying the original list
    include_columns = copy.deepcopy(include_columns) # copy to avoid modifying the original list
    include_recordings = copy.deepcopy(include_recordings) # copy to avoid modifying the original list
    tic = time.time()
    csv_dataset = pd.read_csv(os.path.join(hf_data_dir, csv), nrows=1)
    csv_recording = csv_dataset["recording"].unique()[0]

    if return_masks and "mask_rle" not in include_columns:
        include_columns.append("mask_rle")
    
    if verbose:
        prInfo(f"csv dataset {csv} - {csv_recording}")
    
    if include_recordings != "all":
        if csv_recording not in include_recordings:
            # prWarning(f"Skipping csv {csv} - {csv_recording}")
            return None, csv_recording

    recording_dataset = pd.read_csv(os.path.join(hf_data_dir, csv))
    
    if "ssupaug" or "albee" or "astor" in csv:
        need_sapiens_columns = any([col.startswith("sapiens") for col in include_columns])
        recording_dataset = ssupaug_dataset_handling(recording_dataset, csv_recording, verbose, add_sapiens_columns=need_sapiens_columns)
    
    add_box_columns_for_norm_only = False
    if include_columns != "all":
        # print(f"include_columns: {include_columns}")
        for col in include_columns:
            assert(col in recording_dataset.columns), f"Requested column {col} not found in csv {csv} - {csv_recording}"
            
        # Handle case where bounding box is not in the include columns
        if "xmin" not in include_columns or "xmax" not in include_columns or "ymin" not in include_columns or "ymax" not in include_columns:
            # if one of them is missing assert all of them are 
            assert("xmin" not in include_columns), "If one of xmin, xmax, ymin, ymax is not in include columns, then none of them must be"
            assert("xmax" not in include_columns), "If one of xmin, xmax, ymin, ymax is not in include columns, then none of them must be"
            assert("ymin" not in include_columns), "If one of xmin, xmax, ymin, ymax is not in include columns, then none of them must be"
            assert("ymax" not in include_columns), "If one of xmin, xmax, ymin, ymax is not in include columns, then none of them must be"
            add_box_columns_for_norm_only = True
            include_columns.extend(["xmin", "xmax", "ymin", "ymax"])
            
        if "mask_size" not in include_columns:
            # if mask_size is not in the include columns, then we need to add it eventually rename it to mask_size_meta
            include_columns.append("mask_size")
        
        vitpose_col_to_rename = []
        vitpose_col_to_duplicate = []
        for kpname in VITPOSE_KEYPOINTS_NAMES:
            col = "vitpose_" + kpname + "_score"
            if col not in include_columns:
                # this column is not in the include columns, so we need to rename it to avoid adding it in the data
                vitpose_col_to_rename.append(col)
                include_columns.append(col)
            else:
                # this column is already in the include columns, so we need to duplicate it to keep it in the data
                vitpose_col_to_duplicate.append(col)
                
        recording_dataset = recording_dataset[include_columns]

        if "mask_size" not in original_include_columns:
            # if mask_size is not in the include columns, then we need to rename it to mask_size_meta
            recording_dataset = recording_dataset.rename(columns={"mask_size": "mask_size_meta"})
        else:
            # copy mask_size to mask_size_meta
            recording_dataset["mask_size_meta"] = recording_dataset["mask_size"]
        
        # Create xmin_meta, xmax_meta, ymin_meta, ymax_meta columns for normalization only if needed
        if add_box_columns_for_norm_only:
            # rename xmin, xmax, ymin, ymax to xmin_meta, xmax_meta, ymin_meta, ymax_meta
            recording_dataset = recording_dataset.rename(columns={"xmin": "xmin_meta", "xmax": "xmax_meta", "ymin": "ymin_meta", "ymax": "ymax_meta"})
        else:
            # copy xmin, xmax, ymin, ymax to xmin_meta, xmax_meta, ymin_meta, ymax_meta
            recording_dataset["xmin_meta"] = recording_dataset["xmin"]
            recording_dataset["xmax_meta"] = recording_dataset["xmax"]
            recording_dataset["ymin_meta"] = recording_dataset["ymin"]
            recording_dataset["ymax_meta"] = recording_dataset["ymax"]
            
        for col in vitpose_col_to_rename:
            recording_dataset = recording_dataset.rename(columns={col: col+"_meta"})
            # print(f"[{csv_recording}] renamed {col} to {col}_meta")
        for col in vitpose_col_to_duplicate:
            recording_dataset[col+"_meta"] = recording_dataset[col]
            # print(f"[{csv_recording}] duplicated {col} to {col}_meta")
            
    elif include_columns == "all":
        # duplicate xmin, xmax, ymin, ymax to xmin_meta, xmax_meta, ymin_meta, ymax_meta
        recording_dataset["xmin_meta"] = recording_dataset["xmin"]
        recording_dataset["xmax_meta"] = recording_dataset["xmax"]
        recording_dataset["ymin_meta"] = recording_dataset["ymin"]
        recording_dataset["ymax_meta"] = recording_dataset["ymax"]
    
        #  add mask_size_meta column
        recording_dataset["mask_size_meta"] = recording_dataset["mask_size"]
                
        for kpname in VITPOSE_KEYPOINTS_NAMES:
            col = "vitpose_" + kpname + "_score"
            # duplicate vitpose_score columns to vitpose_score_meta columns
            recording_dataset[col+"_meta"] = recording_dataset[col]

    # Remove mask if not requested to save space
    if not return_masks and "mask_rle" in recording_dataset.columns:
        recording_dataset = recording_dataset.drop(columns=["mask_rle"])
    tac = time.time()
    # prTimer(f"csv dataset {csv} - {csv_recording}", tic, tac)
            
    return recording_dataset, csv_recording


def check_additional_filtering(additional_filtering_dict, track_data: pd.DataFrame) -> np.ndarray[bool]:
    """ Check if the track data is valid according to the additional filtering.
    For now we only support additional filtering as min and max on an existing column.

    Args:
        additional_filtering_dict (dict): additional filtering dictionary
        track_data (pd.DataFrame): track data (extract of the dataset corresponding to a unique track identifier)

    Returns:
        np.ndarray[bool]: mask with length len(track_data)
    """
    
    global_mask = np.ones(len(track_data), dtype=bool)
    for key in additional_filtering_dict:
        if key in track_data.columns:
            key_series = np.array(track_data[key]) # to numpy
            min_mask = np.ones(len(track_data), dtype=bool)
            max_mask = np.ones(len(track_data), dtype=bool)
            if additional_filtering_dict[key]["min"] is not None:
                min_mask = key_series >= additional_filtering_dict[key]["min"]
            if additional_filtering_dict[key]["max"] is not None:
                max_mask = key_series <= additional_filtering_dict[key]["max"]
            
            global_mask[min_mask == False] = False
            global_mask[max_mask == False] = False
            # print(global_mask.min(), global_mask.max())

    return global_mask

def get_keypoints_mask(min_keypoints_filter, track_data: pd.DataFrame, score_threshold: float = 0.5) -> np.ndarray[bool]:
    """ Check if the track data is valid according to the keypoints filtering (number of ViTPose keypoints with score >= score_threshold).

    Args:
        min_keypoints_filter (int): minimum number of keypoints with score >= score_threshold
        track_data (pd.DataFrame): track data (extract of the dataset corresponding to a unique track identifier)
        score_threshold (float, optional): score threshold. Defaults to 0.5.

    Returns:
        np.ndarray[bool]: mask with length len(track_data)
    """
    keypoints_scores = track_data[VITPOSE_KEYPOINTS_SCORES_COLUMNS]
    keypoints_scores_valid = keypoints_scores >= score_threshold
    keypoints_scores_sum = keypoints_scores_valid.sum(axis=1)
    return np.array(keypoints_scores_sum >= min_keypoints_filter)

def get_autovalidity_mask(track_data: pd.DataFrame) -> np.ndarray[bool]:
    """ Check if the track data is valid according to the autovalidity filtering (validity == "valid") established at dataset creation/curation time.

    Args:
        track_data (pd.DataFrame): track data (extract of the dataset corresponding to a unique track identifier)

    Returns:
        np.ndarray[bool]: mask with length len(track_data)
    """
    return np.array(track_data["validity"] == "valid")

def get_existence_mask(track_data: pd.DataFrame) -> tuple[np.ndarray[bool], np.ndarray[int], np.ndarray[int]]:
    """ Check if the track data exists at each step between the first and the last frame of appearance in the recording, this is to identify missing frames and thus remove or split non continuous tracks.

    Args:
        track_data (pd.DataFrame): track data (extract of the dataset corresponding to a unique track identifier)

    Returns:
        np.ndarray[bool]: existence_mask, shape (max_frame_index - min_frame_index + 1,) boolean existence mask indicating if the track data exists at each step
        np.ndarray[int]: existence_mask_index_to_track_data_index, shape (max_frame_index - min_frame_index + 1,) index in existence mask to index in track data, -1 means it does not exist in track_data
        np.ndarray[int]: track_data_index_to_existence_mask_index, shape (len(track_data),) index in track_data to index in existence mask (i.e. referenced between min and max frame index for this track)
    """
    min_frame_index = track_data["image_index"].min()
    max_frame_index = track_data["image_index"].max()
    existence_mask = np.zeros(max_frame_index - min_frame_index + 1, dtype=bool)
    existence_mask_index_to_track_data_index = np.full(max_frame_index - min_frame_index + 1, -1, dtype=int) # index in array to index in track_data
    track_data_index_to_existence_mask_index = np.zeros(len(track_data), dtype=int) # index in track_data to index in array
    for iter, (_, row) in enumerate(track_data.iterrows()):
        existence_mask[row["image_index"] - min_frame_index] = True
        
        track_data_index_to_existence_mask_index[iter] = row["image_index"] - min_frame_index
        existence_mask_index_to_track_data_index[row["image_index"] - min_frame_index] = iter
        
    return existence_mask, existence_mask_index_to_track_data_index, track_data_index_to_existence_mask_index

def get_first_interaction_index(track_data: pd.DataFrame) -> int:
    # index among the track data
    for iter, (index, row) in enumerate(track_data.iterrows()):
        if row["engagement"] > 0:
            return iter
    return -1 

def get_biggest_mask_index(track_data: pd.DataFrame) -> int:
    """ Get the index of the biggest mask size in the track data.
    """
    masks_size_series = np.array(track_data["mask_size_meta"]) # use the meta so that even if the mask_size column is not present, we can still get the biggest mask size
    return np.argmax(masks_size_series)

def get_track_input_possible_indices(track_data: pd.DataFrame, 
                                     unique_track_identifier: str, 
                                     input_length_in_frames: int, 
                                     fixed_input_length: bool, 
                                     min_length_in_frames: int, 
                                     max_length_in_frames: int, 
                                     interaction_cutoff: int,
                                     positive_cutoff: int,
                                     force_positive_samples: bool,
                                     force_aligment_with_biggest_mask_size: bool,
                                     additional_filtering_dict: dict, 
                                     min_keypoints_filter: int,
                                     center_on_onset: bool = False,
                                     cutoffs_filtering: bool = True) -> tuple[int, list[tuple[int, int, int]]]:
    
    """ Get the possible input indices for a track

    Args:
        track_data (pd.DataFrame): track data (extract of the dataset corresponding to a unique track identifier)
        unique_track_identifier (str): unique track identifier
        input_length_in_frames (int): input length in frames
        fixed_input_length (bool): if True, the input length is fixed
        min_length_in_frames (int): minimum length in frames
        max_length_in_frames (int): maximum length in frames
        interaction_cutoff (int): interaction cutoff (corresponds to T_CUT)
        positive_cutoff (int): positive cutoff (corresponds to T_INT)
        force_positive_samples (bool): if True, force positive samples
        force_aligment_with_biggest_mask_size (bool): if True, force alignment with the biggest mask size
        additional_filtering_dict (dict): additional filtering dictionary
        min_keypoints_filter (int): minimum number of keypoints with score >= score_threshold
        center_on_onset (bool): if True, center the input on the onset of the interaction (useful for display purposes)
        cutoffs_filtering (bool): if True, filter on cutoffs, interaction etc... without it take all possible segments, always considered false label (for pretraining)
    Returns:
        tuple[int, list[tuple[int, int, int]]]: first interaction index, list of tuples (possible_start_frame, number_of_frames, label)
    """
    
    has_vitpose_keypoints_scores = all([col in track_data.columns for col in VITPOSE_KEYPOINTS_SCORES_COLUMNS])

    min_frames = input_length_in_frames if fixed_input_length else min_length_in_frames
    assert(type(min_frames) == int), "min_length_in_frames must be an integer"
        
    if len(track_data) < min_frames:
        return None, []
    
    additional_filtering_mask = check_additional_filtering(additional_filtering_dict, track_data) # mask with length len(track_data)
    if has_vitpose_keypoints_scores:
        keypoints_mask = get_keypoints_mask(min_keypoints_filter, track_data) # mask with length len(track_data)
    else:
        keypoints_mask = np.ones(len(track_data), dtype=bool)
    autovalidity_mask = get_autovalidity_mask(track_data) # mask with length len(track_data)
    
    existence_mask, existence_mask_index_to_track_data_index, track_data_index_to_existence_mask_index = get_existence_mask(track_data) # existence mask as length max_frame_index - min_frame_index + 1
    
    full_validity_mask = np.zeros((4, existence_mask.shape[0]), dtype=bool) # mask with length max_frame_index - min_frame_index + 1, 4 channels of mask : additional, keypoints, autovalidity, existence
    
    full_validity_mask[0, :][track_data_index_to_existence_mask_index] = additional_filtering_mask
    full_validity_mask[1, :][track_data_index_to_existence_mask_index] = keypoints_mask
    full_validity_mask[2, :][track_data_index_to_existence_mask_index] = autovalidity_mask
    full_validity_mask[3] = existence_mask
            
    full_validity_mask = np.logical_and.reduce(full_validity_mask, axis=0) # shape (max_frame_index - min_frame_index + 1,) if one of the channels is False, the track is not valid at this time
    
    first_interaction_index = get_first_interaction_index(track_data) # index among track_data
    
    
    if not cutoffs_filtering:
        assert(fixed_input_length), "disabling cutoffs filtering is only implemented for fixed input length"
        # no filtering on cutoffs, interaction etc... let every contiguous segment of the right length go its way
        if full_validity_mask.sum() == 0:
            return None, []
        
        contiguous_true_segments = get_contiguous_true_segments(full_validity_mask) # list of tuples (start, length) with the positions of the true segments and their lengths. Inclusive of the first element. Positions in image indexes relatively to the first image index of the track
        possible_indices = []
        for (start, length) in contiguous_true_segments:
            if length >= min_frames:
                start_in_track_data_index = existence_mask_index_to_track_data_index[start]
                possible_starting_points = range(start_in_track_data_index, start_in_track_data_index + length + 1 - min_frames) # such that the end of the input is still in the valid segment. 
                for possible_starting_point in possible_starting_points:
                    label = 0
                    possible_indices.append((possible_starting_point, input_length_in_frames, label))
                    
        return first_interaction_index, possible_indices
                    

    if first_interaction_index != -1:
        # has an interaction
        
        interaction_index_in_existence_mask = track_data_index_to_existence_mask_index[first_interaction_index]

        if center_on_onset:
            # when doing this it will result in a single possible segment because of the length constraint !
            start_min = max(interaction_index_in_existence_mask-(input_length_in_frames//2), 0)
            if input_length_in_frames%2 == 1:
                end_max = min(interaction_index_in_existence_mask+(input_length_in_frames//2)+1, len(full_validity_mask)-1)
            else:
                end_max = min(interaction_index_in_existence_mask+(input_length_in_frames//2), len(full_validity_mask)-1)
            full_validity_mask[end_max:] = False 
            full_validity_mask[:start_min] = False 
            min_frames_in_segment = min_frames
        else:
            # normal behavior
            full_validity_mask[interaction_index_in_existence_mask+1:] = False # allow to go up to the start of the interaction
            min_frames_in_segment = min_frames + interaction_cutoff # we need to cut interaction_cutoff frames before the first interaction
            
        biggest_mask_index_in_existence_mask = None
        
    else:
        # no interaction
        
        biggest_mask_index = get_biggest_mask_index(track_data)
        biggest_mask_index_in_existence_mask = track_data_index_to_existence_mask_index[biggest_mask_index]
        
        if center_on_onset:
            # when doing this it will result in a single possible segment because of the length constraint !
            start_min = max(biggest_mask_index_in_existence_mask-(input_length_in_frames//2), 0)
            if input_length_in_frames%2 == 1:
                end_max = min(biggest_mask_index_in_existence_mask+(input_length_in_frames//2)+1, len(full_validity_mask)-1)
            else:
                end_max = min(biggest_mask_index_in_existence_mask+(input_length_in_frames//2), len(full_validity_mask)-1)
            full_validity_mask[end_max:] = False
            full_validity_mask[:start_min] = False
            min_frames_in_segment = min_frames            
        elif force_aligment_with_biggest_mask_size:
            # only consider the part before the biggest mask size
            full_validity_mask[biggest_mask_index_in_existence_mask+1:] = False
            min_frames_in_segment = min_frames + interaction_cutoff # make sure there is enough frame to cut interaction_cutoff before the onset
        else:
            # completetely random choice
            biggest_mask_index_in_existence_mask = None
            min_frames_in_segment = min_frames

        interaction_index_in_existence_mask = None


    if full_validity_mask.sum() == 0:
        return None, []
    
        
    contiguous_true_segments = get_contiguous_true_segments(full_validity_mask) # list of tuples (start, length) with the positions of the true segments and their lengths. Inclusive of the first element. Positions in image indexes relatively to the first image index of the track
    possible_indices = []
    for (start, length) in contiguous_true_segments:
        if length >= min_frames_in_segment:
            start_in_track_data_index = existence_mask_index_to_track_data_index[start]
            assert(start_in_track_data_index != -1), f"Start in track data index is -1 for track {unique_track_identifier} at index {start} it should not happen"
            
            if fixed_input_length:
                possible_starting_points = range(start_in_track_data_index, start_in_track_data_index + length + 1 - min_frames_in_segment) # such that the end of the input is still in the valid segment. 
                # Example with start_in_track_data_index = 100, input_length_in_frames = min_frames = 30, interaction_cutoff = 5, length = 50
                # If not interaction we have range(100, 100+50-30) = range(100, 120)
                # If interaction we have range(100, 100+50-35) = range(100, 115) such that the end of the input of length (min_frames = input_length_in_frames) is at least interaction_cutoff frames before the interaction
                for possible_starting_point in possible_starting_points:
                    if first_interaction_index != -1:
                        if possible_starting_point + input_length_in_frames >= first_interaction_index - positive_cutoff + 1:
                            label = 1 # interaction if the the input ends at least positive_cutoff frames before the interaction
                        else:
                            label = 0 # ending too early, consider it as no interaction
                            if force_positive_samples:
                                # dont allow for negative samples when possible to force positive samples
                                continue
                    else:
                        label = 0
                        if center_on_onset:
                            # for display purposes with center_on_onset, allow all points
                            pass
                        elif force_aligment_with_biggest_mask_size and possible_starting_point + input_length_in_frames != biggest_mask_index_in_existence_mask - interaction_cutoff + 1:
                            # force to end right on the biggest mask size
                            continue
                        else:
                            # normal case but not forcing alignment
                            pass
                        
                    possible_indices.append((possible_starting_point, input_length_in_frames, label))
                    
            else:
                possible_starting_points = range(start_in_track_data_index, start_in_track_data_index + length - min_frames_in_segment) # such that the end of the input is still in the valid segment
                for seg_count, possible_starting_point in enumerate(possible_starting_points):
                    # Example with start_in_track_data_index = 100, min_length_in_frames = min_frames = 20, max_length_in_frames = 40, interaction_cutoff = 5, length = 50
                    # If not interaction we have possible_starting_points = range(100, 100+50-20+1) = range(100, 130)
                    # Example with seg_count = 5, possible_starting_point = 105
                    max_frames_for_this_starting_point = min(max_length_in_frames, length - seg_count) # Example with seg_count = 5, possible_starting_point = 105 --> min(40, 50-5) = 40 | Example with seg_count = 17, possible_starting_point = 117 --> min(40, 50-17) = 33
                    for possible_frame_count in range(min_length_in_frames, max_frames_for_this_starting_point + 1):
                        if first_interaction_index != -1:
                            if possible_starting_point + possible_frame_count >= first_interaction_index - positive_cutoff + 1:
                                label = 1 # interaction if the the input ends at least positive_cutoff frames before the interaction
                            else:
                                label = 0
                                if force_positive_samples:
                                    # dont allow for negative samples when possible to force positive samples
                                    continue
                        else:
                            # negative sample anyway
                            label = 0
                            
                        possible_indices.append((possible_starting_point, possible_frame_count, label))
    
    return first_interaction_index, possible_indices


def process_track_input(args: dict) -> tuple[str, int, list[tuple[int, int, int]]]:
    
    unique_track_identifier = args["unique_track_identifier"]
    track_data = args["track_data"]
    input_length_in_frames = args["input_length_in_frames"]
    fixed_input_length = args["fixed_input_length"]
    min_length_in_frames = args["min_length_in_frames"]
    max_length_in_frames = args["max_length_in_frames"]
    interaction_cutoff = args["interaction_cutoff"]
    positive_cutoff = args["positive_cutoff"]
    force_positive_samples = args["force_positive_samples"]
    additional_filtering_dict = args["additional_filtering_dict"]
    min_keypoints_filter = args["min_keypoints_filter"]
    force_aligment_with_biggest_mask_size = args["force_aligment_with_biggest_mask_size"]
    center_on_onset = args["center_on_onset"]
    cutoffs_filtering = args["cutoffs_filtering"] # by default this should always be True
    
    first_interaction_index, possible_indices = get_track_input_possible_indices(track_data, 
                                                                                 unique_track_identifier, 
                                                                                 input_length_in_frames, 
                                                                                 fixed_input_length, 
                                                                                 min_length_in_frames, 
                                                                                 max_length_in_frames, 
                                                                                 interaction_cutoff, 
                                                                                 positive_cutoff, 
                                                                                 force_positive_samples, 
                                                                                 force_aligment_with_biggest_mask_size,
                                                                                 additional_filtering_dict, 
                                                                                 min_keypoints_filter,
                                                                                 center_on_onset,
                                                                                 cutoffs_filtering)
    return (unique_track_identifier, first_interaction_index, possible_indices)


def apply_dataset_rescaling_rules(input_tensor, metadata, dataset_rescaling_rules):
    """ Apply the dataset rescaling rules to the input tensor.

    Args:
        input_tensor (torch.Tensor): input tensor of shape (B, T, V, C)
        metadata (list): metadata list of dictionaries
        dataset_rescaling_rules (dict): dataset rescaling rules dictionary

    Returns:
        torch.Tensor: input tensor of shape (B, T, V, C)
    """
    B, T, V, C = input_tensor.shape

    hui360_orig_mask = [True if meta["orig_dataset"] == "HUI360" else False for meta in metadata]
    hui360_orig_mask = torch.tensor(hui360_orig_mask, dtype=torch.bool) #, device=input_tensor.device)
    
    amass_orig_mask = [True if meta["orig_dataset"] == "AMASS" else False for meta in metadata]
    amass_orig_mask = torch.tensor(amass_orig_mask, dtype=torch.bool) #, device=input_tensor.device)
    input_tensor[amass_orig_mask, :, :, 2] = 1.0 # replace depth by 1.0 confidence for amass
    
    h36msh_orig_mask = [True if meta["orig_dataset"] == "H36M-SH" else False for meta in metadata]
    h36msh_orig_mask = torch.tensor(h36msh_orig_mask, dtype=torch.bool) #, device=input_tensor.device)
    input_tensor[h36msh_orig_mask, :, :, 2] = 1.0 # replace depth by 1.0 confidence for h36m-sh
            
    posetrack_orig_mask = [True if meta["orig_dataset"] == "PoseTrack18" else False for meta in metadata]
    posetrack_orig_mask = torch.tensor(posetrack_orig_mask, dtype=torch.bool) #, device=input_tensor.device)
    
    instavariety_orig_mask = [True if meta["orig_dataset"] == "InstaVariety" else False for meta in metadata]
    instavariety_orig_mask = torch.tensor(instavariety_orig_mask, dtype=torch.bool) #, device=input_tensor.device)
    
    jrdb_orig_mask = [True if meta["orig_dataset"] == "JRDB" else False for meta in metadata]
    jrdb_orig_mask = torch.tensor(jrdb_orig_mask, dtype=torch.bool) #, device=input_tensor.device)
    input_tensor[jrdb_orig_mask, :, :, 2] = input_tensor[jrdb_orig_mask, :, :, 2] / 2.0 # normalize scores to be in 0-1 (originally 0 = invisible, 1 = occluded, 2 = visible)

    posesinthewild_orig_mask = [True if meta["orig_dataset"] == "PosesInTheWild" else False for meta in metadata]
    posesinthewild_orig_mask = torch.tensor(posesinthewild_orig_mask, dtype=torch.bool) #, device=input_tensor.device)

    if hui360_orig_mask.sum() > 0:
        # orig_dataset = ["HUI360"]
        input_tensor[hui360_orig_mask,...] = coco2h36m(input_tensor[hui360_orig_mask,...])
        w, h = dataset_rescaling_rules["HUI360"]["normalize_image"]
        # scale = min(w,h) / 2.0
        scale_image = torch.tensor([w, h], device=input_tensor.device) / 2.0
        input_tensor[hui360_orig_mask,...,:2] = input_tensor[hui360_orig_mask,...,:2] - scale_image
        input_tensor[hui360_orig_mask,...,:2] = input_tensor[hui360_orig_mask,...,:2] / scale_image
        # input_tensor = input_tensor.cuda() # B, T, V, C
        
    if jrdb_orig_mask.sum() > 0:
        # pass
        w, h = dataset_rescaling_rules["JRDB"]["normalize_image"]
        # # scale = min(w,h) / 2.0
        scale_image = torch.tensor([w, h], device=input_tensor.device) / 2.0
        input_tensor[jrdb_orig_mask,...,:2] = input_tensor[jrdb_orig_mask,...,:2] - scale_image
        input_tensor[jrdb_orig_mask,...,:2] = input_tensor[jrdb_orig_mask,...,:2] / scale_image
        
    if posesinthewild_orig_mask.sum() > 0:
        # pass
        w, h = dataset_rescaling_rules["PosesInTheWild"]["normalize_image"]
        # # scale = min(w,h) / 2.0
        scale_image = torch.tensor([w, h], device=input_tensor.device) / 2.0
        input_tensor[posesinthewild_orig_mask,...,:2] = input_tensor[posesinthewild_orig_mask,...,:2] - scale_image
        input_tensor[posesinthewild_orig_mask,...,:2] = input_tensor[posesinthewild_orig_mask,...,:2] / scale_image
    
    if dataset_rescaling_rules["track_normalize"]:
        # ignore invalid keypoints
        invalid_keypoints_mask = input_tensor[...,2] < 0.25 # shape (B, T, V)
        input_tensor_for_min = input_tensor[...,:2].clone()
        input_tensor_for_min[invalid_keypoints_mask] = float('inf')
        input_tensor_for_max = input_tensor[...,:2].clone()
        input_tensor_for_max[invalid_keypoints_mask] = float('-inf')
        
        min_xy_per_sample = input_tensor_for_min[...,:2].amin(dim=(1,2)) # shape (B, 2)
        max_xy_per_sample = input_tensor_for_max[...,:2].amax(dim=(1,2)) # shape (B, 2)
        
        range_xy_per_sample = max_xy_per_sample - min_xy_per_sample # shape (B, 2)

        input_tensor[...,:2] = (input_tensor[...,:2] - min_xy_per_sample.unsqueeze(1).unsqueeze(2)) / range_xy_per_sample.unsqueeze(1).unsqueeze(2)
        input_tensor[...,:2] = input_tensor[...,:2] * 2.0 - 1.0 # between -1 and 1

        if torch.isnan(input_tensor).any():
            raise ValueError(f"Input tensor is nan after rescaling (B,T,V,C = {B},{T},{V},{C}, orig of samples : {[meta['orig_dataset'] for meta in metadata]})")
    
    else:
        rescaling_tensor = torch.ones((B,1,1,2), device=input_tensor.device)
        rescaling_tensor[hui360_orig_mask, ...]  = torch.tensor(dataset_rescaling_rules["HUI360"]["scale_factor"], device=input_tensor.device)
        rescaling_tensor[amass_orig_mask, ...] = torch.tensor(dataset_rescaling_rules["AMASS"]["scale_factor"], device=input_tensor.device)
        rescaling_tensor[h36msh_orig_mask, ...] = torch.tensor(dataset_rescaling_rules["H36M-SH"]["scale_factor"], device=input_tensor.device)
        rescaling_tensor[posetrack_orig_mask, ...] = torch.tensor(dataset_rescaling_rules["PoseTrack18"]["scale_factor"], device=input_tensor.device)
        rescaling_tensor[instavariety_orig_mask, ...] = torch.tensor(dataset_rescaling_rules["InstaVariety"]["scale_factor"], device=input_tensor.device)
        rescaling_tensor[jrdb_orig_mask, ...] = torch.tensor(dataset_rescaling_rules["JRDB"]["scale_factor"], device=input_tensor.device)
        rescaling_tensor[posesinthewild_orig_mask, ...] = torch.tensor(dataset_rescaling_rules["PosesInTheWild"]["scale_factor"], device=input_tensor.device)
        rescaling_tensor = rescaling_tensor.repeat(1,T,V,1)
            
        shift_tensor = torch.zeros((B,1,1,2), device=input_tensor.device)
        shift_tensor[hui360_orig_mask, ...] = torch.tensor(dataset_rescaling_rules["HUI360"]["fix_shift"], device=input_tensor.device)
        shift_tensor[amass_orig_mask, ...] = torch.tensor(dataset_rescaling_rules["AMASS"]["fix_shift"], device=input_tensor.device)
        shift_tensor[h36msh_orig_mask, ...] = torch.tensor(dataset_rescaling_rules["H36M-SH"]["fix_shift"], device=input_tensor.device)
        shift_tensor[posetrack_orig_mask, ...] = torch.tensor(dataset_rescaling_rules["PoseTrack18"]["fix_shift"], device=input_tensor.device)
        shift_tensor[instavariety_orig_mask, ...] = torch.tensor(dataset_rescaling_rules["InstaVariety"]["fix_shift"], device=input_tensor.device)
        shift_tensor[jrdb_orig_mask, ...] = torch.tensor(dataset_rescaling_rules["JRDB"]["fix_shift"], device=input_tensor.device)
        shift_tensor[posesinthewild_orig_mask, ...] = torch.tensor(dataset_rescaling_rules["PosesInTheWild"]["fix_shift"], device=input_tensor.device)
        shift_tensor = shift_tensor.repeat(1,T,V,1)
    
        input_tensor[...,:2] = input_tensor[...,:2] * rescaling_tensor + shift_tensor
    
    return input_tensor
        