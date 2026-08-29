# Import standard libraries
import argparse
from math import e
import os
import sys
# Define the here variable to be the directory of the current file
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, ".."))
from tqdm import tqdm
import joblib
from collections import OrderedDict
import socket
hostname = socket.gethostname()
import warnings
warnings.filterwarnings("ignore")

# # Import torch libraries
# import torch
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import torch.nn as nn
# import torch.multiprocessing as mp
# mp.set_sharing_strategy("file_system")

# Import random library
import random
import numpy as np

# import metrics and classifiers from scikit-learn
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, average_precision_score

from datasets.HUIDatasetUtils import (
    input_tensor_to_format_by_channel, 
    input_tensor_to_format_by_channel_sapiens_without_face,
    keypoints17_to_coco18, 
    coco2h36m, 
    crop_scale_torch, 
    crop_scale_torch_by_sample, 
    convert_to_polar_coordinates,
    keypoints17_to_coco18_torch,
    coco2nturgbd,
    coco2nturgbd_nospine_mid,
    coco2nwucla,
    sapiensnoface2nturgbd_nospine_mid
)

# Import custom utils
from utils.print_utils import *
from utils.loader_utils import load_hui_dataset
from utils.data_utils import VITPOSE_KEYPOINTS_NAMES
from utils.other_utils import read_yaml_to_dic
from utils.debug_utils import plot_input_tensor, update_old_config_dict, plot_input_tensors_skformer, export_unique_track_identifiers
from utils.eval_utils import get_best_threshold_f1


def extract_dataset_to_dict(dataset, config):
    n = len(dataset)
    data = np.empty((n, 48, 17, 3), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    ids = np.empty(n, dtype=object)

    for i, (input_tensor, label, metadata_dict, _images_tensor, _masks_tensor) in enumerate(
        tqdm(dataset, total=n, desc="Extracting samples")
    ):
        if hasattr(input_tensor, "numpy"):
            data[i] = input_tensor.numpy()
        else:
            data[i] = np.asarray(input_tensor, dtype=np.float32)

        if hasattr(label, "item"):
            labels[i] = int(label.item())
        else:
            labels[i] = int(label)

        ids[i] = metadata_dict["unique_track_identifier"]

    return {"config": config, "data": data, "labels": labels, "ids": ids}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract the data from HUI train, HUI test, SSUP train, SSUP test')
    # parser.add_argument("--model_path", "-m", required=True, type=str, help="Path to the .pth model checkpoint file. Required.")
    parser.add_argument("--num_workers", default=None, type=int, help="Number of workers for dataloader (default: auto)")
    parser.add_argument("--hf_local_dir", "-hld", default="default", type=str, help="HF local directory (default: default, ie ./datasets/hf_data)")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Verbose mode")
    parser.add_argument("--output_dir", "-o", default=os.path.join(here, "datasets", "sk_extracted"), type=str, help="Directory for exported pickle files")
    parser.add_argument("--offline_mode", "-om", action="store_true", default=False, help="Offline mode, do not download from Hugging Face")
    args = parser.parse_args()
    args.preload_data = False
    args.preload_only = False
    
    
    default_config = {'train_tracks_filename': 'all',
                      'val_tracks_filename': 'all',
                      'test_tracks_filename': 'all',
                      'positive_cutoff_train': 0,
                      'interaction_cutoff_train': 0,
                      'positive_cutoff_val': 0, 
                      'interaction_cutoff_val': 0,
                      'fixed_input_length': True,
                      'input_length_in_frames': 48,
                      'subsample_frames': 1,
                      'fix_index_per_track_train': True,
                      'fix_index_per_track_list_train': [3662, 5427, 3606, 3726, 3417, 6031, 7527, 1501, 4501, 9588, 2712, 4509, 2752, 57, 9256, 3417, 8694, 9336, 6870, 3587, 2675, 3613, 9281, 4883, 7570, 8967, 1654, 5194, 9746, 4310, 2848, 9954],
                      'fix_index_per_track_val': True,
                      'fix_index_per_track_list_val': [3662, 5427, 3606, 3726, 3417, 6031, 7527, 1501, 4501, 9588, 2712, 4509, 2752, 57, 9256, 3417, 8694, 9336, 6870, 3587, 2675, 3613, 9281, 4883, 7570, 8967, 1654, 5194, 9746, 4310, 2848, 9954],
                      'min_length_in_frames': None,
                      'max_length_in_frames': None,
                      'min_keypoints_filter': 9,
                      'additional_filtering_dict': {'mask_size': {'min': 1000, 'max': 10000000.0}},
                      'normalize_in_image': False, 
                      'normalize_keypoints_in_box': False,
                      'normalize_keypoints_in_track': 'none',
                      'do_recenter_interaction_zone': True,
                      'standardize_data': 'none',
                      'use_polar_coordinates': False,
                      'force_positive_samples': True,
                      'force_aligment_with_biggest_mask_size_train': True,
                      'force_aligment_with_biggest_mask_size_val': True,
                      'center_on_onset_train': False,
                      'center_on_onset_val': False,
                      'random_flip_horizontal_train': False,
                      'random_flip_horizontal_val': False,
                      'random_jitter_position_train': [0.0, 0.0],
                      'random_jitter_position_val': [0.0, 0.0],
                      'do_recentering_train': False,
                      'do_recentering_val': False,
                      'do_fix_keypoints_outside_box_train': True,
                      'do_fix_keypoints_outside_box_val': True,
                      'inputs_per_track_stride_train': -1,
                      'inputs_per_track_stride_val': -1,
                      # 'include_recordings_train': ['rosbag2_2025_07_10-10_29_13', 'rosbag2_2025_07_10-15_47_18', 'rosbag2_2025_07_07-10_24_20', 'rosbag2_2025_07_16-13_12_03', 'rosbag2_2025_07_16-14_07_49', 'rosbag2_2025_07_16-15_50_45', 'rosbag2_2025_07_23-11_01_56', 'rosbag2_2025_07_23-12_18_45', 'rosbag2_2025_07_23-13_17_40', 'rosbag2_2025_07_23-14_30_55', 'rosbag2_2025_07_29-10_23_14', 'rosbag2_2025_07_29-13_17_18', 'rosbag2_2025_07_29-14_09_46', 'rosbag2_2025_10_15-12_02_00', 'rosbag2_2025_10_15-13_11_27', 'rosbag2_2025_10_15-14_02_29', 'rosbag2_2025_10_15-14_30_03', 'rosbag2_2025_10_16-09_27_48', 'rosbag2_2025_10_16-12_47_57', 'rosbag2_2025_10_20-10_13_32', 'rosbag2_2025_10_20-11_51_02', 'rosbag2_2025_10_20-16_02_46', 'rosbag2_2025_07_11-10_28_07', 'rosbag2_2025_07_11-11_18_00', 'rosbag2_2025_07_18-10_37_07', 'rosbag2_2025_07_25-10_52_32', 'rosbag2_2025_07_25-14_14_16', 'rosbag2_2025_10_09-08_52_20', 'rosbag2_2025_07_11-13_27_26', 'rosbag2_2025_07_11-14_54_55', 'rosbag2_2025_07_17-11_28_34', 'rosbag2_2025_07_17-12_52_12', 'rosbag2_2025_07_24-10_41_01', 'rosbag2_2025_07_24-12_14_56', 'rosbag2_2025_07_24-13_33_54', 'rosbag2_2025_07_24-14_33_36', 'rosbag2_2025_07_07-10_49_31', 'rosbag2_2025_07_22-09_38_18', 'rosbag2_2025_07_22-10_59_25', 'rosbag2_2025_07_22-12_18_30', 'rosbag2_2025_07_22-13_30_39', 'rosbag2_2025_10_09-10_23_38', 'rosbag2_2025_10_15-09_37_49', 'rosbag2_2025_10_15-11_03_05'], 
                      # 'include_recordings_val': ['rosbag2_2025_07_07-11_16_10', 'rosbag2_2025_07_07-12_38_45', 'rosbag2_2025_07_07-15_33_32', 'rosbag2_2025_07_15-12_39_21', 'rosbag2_2025_07_15-13_41_01', 'rosbag2_2025_07_15-14_48_22', 'rosbag2_2025_07_21-10_22_11', 'rosbag2_2025_07_21-11_56_40', 'rosbag2_2025_07_21-13_09_22', 'rosbag2_2025_07_21-14_11_37', 'rosbag2_2025_07_21-15_15_07', 'rosbag2_2025_07_28-10_18_10', 'rosbag2_2025_07_28-11_25_33', 'rosbag2_2025_07_28-13_05_46', 'rosbag2_2025_07_28-14_19_07', 'rosbag2_2025_10_17-13_19_29', 'rosbag2_2025_10_17-14_28_15', 'rosbag2_2025_10_17-15_11_00', 'rosbag2_2025_10_17-16_47_09', 'rosbag2_2025_10_07-15_03_48', 'rosbag2_2025_10_07-16_21_39', 'rosbag2_2025_10_09-17_37_23', 'rosbag2_2025_10_09-18_50_21', 'rosbag2_2025_10_15-12_27_14', 'rosbag2_2025_10_16-11_29_56'],
                      'include_columns': ['recording', 'episode', 'image_height', 'image_width', 'unique_track_identifier', 'track_id', 'image_file', 'image_index', 'validity', 'current_segment', 'total_segments', 'position_in_segment', 'length_of_current_segment', 'timestamp', 'timestamp_sec', 'timestamp_track', 'engagement', 'time_to_first_interaction', 'mask_size', 'xmin', 'ymin', 'xmax', 'ymax', 'vitpose_nose_x', 'vitpose_nose_y', 'vitpose_nose_score', 'vitpose_left_eye_x', 'vitpose_left_eye_y', 'vitpose_left_eye_score', 'vitpose_right_eye_x', 'vitpose_right_eye_y', 'vitpose_right_eye_score', 'vitpose_left_ear_x', 'vitpose_left_ear_y', 'vitpose_left_ear_score', 'vitpose_right_ear_x', 'vitpose_right_ear_y', 'vitpose_right_ear_score', 'vitpose_left_shoulder_x', 'vitpose_left_shoulder_y', 'vitpose_left_shoulder_score', 'vitpose_right_shoulder_x', 'vitpose_right_shoulder_y', 'vitpose_right_shoulder_score', 'vitpose_left_elbow_x', 'vitpose_left_elbow_y', 'vitpose_left_elbow_score', 'vitpose_right_elbow_x', 'vitpose_right_elbow_y', 'vitpose_right_elbow_score', 'vitpose_left_wrist_x', 'vitpose_left_wrist_y', 'vitpose_left_wrist_score', 'vitpose_right_wrist_x', 'vitpose_right_wrist_y', 'vitpose_right_wrist_score', 'vitpose_left_hip_x', 'vitpose_left_hip_y', 'vitpose_left_hip_score', 'vitpose_right_hip_x', 'vitpose_right_hip_y', 'vitpose_right_hip_score', 'vitpose_left_knee_x', 'vitpose_left_knee_y', 'vitpose_left_knee_score', 'vitpose_right_knee_x', 'vitpose_right_knee_y', 'vitpose_right_knee_score', 'vitpose_left_ankle_x', 'vitpose_left_ankle_y', 'vitpose_left_ankle_score', 'vitpose_right_ankle_x', 'vitpose_right_ankle_y', 'vitpose_right_ankle_score'],
                      'hf_dataset_revision': 'main',
                      'comment': 'NoComment',
                      'force_align_negatives_train': 'force_aligment',
                      'force_align_negatives_val': 'force_aligment',
                    #   'cross_eval_type': 'hui_train_vs_hui_test',
                    #   'valid': True,
                    #   'hostname': 'raphael-Precision-3591',
                    #   'experiment_name': 'in_hui_lstm',
                    #   'remove_joints': None,
                    #   'perspective_reprojection': {'do_perspective_reprojection': False, 'hcenter': 0.0, 'vcenter': 0.0, 'hfov': 94.0, 'vfov': 68.0, 'target_width': 1280},
                      'format_by_channel': True,
                    #   'mb_desired_return': 'representation',
                      'cutoffs_filtering': True}


    hui_config = default_config.copy()
    hui_config["include_recordings_train"] = ['rosbag2_2025_07_10-10_29_13', 'rosbag2_2025_07_10-15_47_18', 'rosbag2_2025_07_07-10_24_20', 'rosbag2_2025_07_16-13_12_03', 'rosbag2_2025_07_16-14_07_49', 'rosbag2_2025_07_16-15_50_45', 'rosbag2_2025_07_23-11_01_56', 'rosbag2_2025_07_23-12_18_45', 'rosbag2_2025_07_23-13_17_40', 'rosbag2_2025_07_23-14_30_55', 'rosbag2_2025_07_29-10_23_14', 'rosbag2_2025_07_29-13_17_18', 'rosbag2_2025_07_29-14_09_46', 'rosbag2_2025_10_15-12_02_00', 'rosbag2_2025_10_15-13_11_27', 'rosbag2_2025_10_15-14_02_29', 'rosbag2_2025_10_15-14_30_03', 'rosbag2_2025_10_16-09_27_48', 'rosbag2_2025_10_16-12_47_57', 'rosbag2_2025_10_20-10_13_32', 'rosbag2_2025_10_20-11_51_02', 'rosbag2_2025_10_20-16_02_46', 'rosbag2_2025_07_11-10_28_07', 'rosbag2_2025_07_11-11_18_00', 'rosbag2_2025_07_18-10_37_07', 'rosbag2_2025_07_25-10_52_32', 'rosbag2_2025_07_25-14_14_16', 'rosbag2_2025_10_09-08_52_20', 'rosbag2_2025_07_11-13_27_26', 'rosbag2_2025_07_11-14_54_55', 'rosbag2_2025_07_17-11_28_34', 'rosbag2_2025_07_17-12_52_12', 'rosbag2_2025_07_24-10_41_01', 'rosbag2_2025_07_24-12_14_56', 'rosbag2_2025_07_24-13_33_54', 'rosbag2_2025_07_24-14_33_36', 'rosbag2_2025_07_07-10_49_31', 'rosbag2_2025_07_22-09_38_18', 'rosbag2_2025_07_22-10_59_25', 'rosbag2_2025_07_22-12_18_30', 'rosbag2_2025_07_22-13_30_39', 'rosbag2_2025_10_09-10_23_38', 'rosbag2_2025_10_15-09_37_49', 'rosbag2_2025_10_15-11_03_05']
    hui_config["include_recordings_val"] = ['rosbag2_2025_07_07-11_16_10', 'rosbag2_2025_07_07-12_38_45', 'rosbag2_2025_07_07-15_33_32', 'rosbag2_2025_07_15-12_39_21', 'rosbag2_2025_07_15-13_41_01', 'rosbag2_2025_07_15-14_48_22', 'rosbag2_2025_07_21-10_22_11', 'rosbag2_2025_07_21-11_56_40', 'rosbag2_2025_07_21-13_09_22', 'rosbag2_2025_07_21-14_11_37', 'rosbag2_2025_07_21-15_15_07', 'rosbag2_2025_07_28-10_18_10', 'rosbag2_2025_07_28-11_25_33', 'rosbag2_2025_07_28-13_05_46', 'rosbag2_2025_07_28-14_19_07', 'rosbag2_2025_10_17-13_19_29', 'rosbag2_2025_10_17-14_28_15', 'rosbag2_2025_10_17-15_11_00', 'rosbag2_2025_10_17-16_47_09', 'rosbag2_2025_10_07-15_03_48', 'rosbag2_2025_10_07-16_21_39', 'rosbag2_2025_10_09-17_37_23', 'rosbag2_2025_10_09-18_50_21', 'rosbag2_2025_10_15-12_27_14', 'rosbag2_2025_10_16-11_29_56']

    ssup_config = default_config.copy()
    ssup_config["include_recordings_train"] = ["2022_09_21_astor_place_landfill","2022_09_21_astor_place_recycle","2022_09_26_astor_place_landfill","2022_09_26_astor_place_recycle","2022_09_28_astor_place_landfill","2022_09_28_astor_place_recycle","2022_10_06_astor_place_landfill","2022_10_06_astor_place_recycle","2022_10_12_astor_place_landfill_0","2022_10_12_astor_place_landfill_1","2022_10_12_astor_place_recycle_0","2022_10_12_astor_place_recycle_1"]
    ssup_config["include_recordings_val"] = ["2023_07_06_albee_square_landfill_0","2023_07_06_albee_square_landfill_1","2023_07_06_albee_square_recycle_0","2023_07_06_albee_square_recycle_1","2023_07_07_albee_square_landfill_0","2023_07_07_albee_square_landfill_1","2023_07_07_albee_square_recycle_0","2023_07_07_albee_square_recycle_1","2023_07_11_albee_square_landfill_0","2023_07_11_albee_square_landfill_1","2023_07_11_albee_square_recycle_0","2023_07_11_albee_square_recycle_1","2023_07_12_albee_square_landfill_0","2023_07_12_albee_square_landfill_1","2023_07_12_albee_square_recycle_0","2023_07_12_albee_square_recycle_1","2023_07_14_albee_square_landfill","2023_07_14_albee_square_recycle"]

    hui_train_dataset = load_hui_dataset(args, hui_config, split="train", num_workers=0) # expected 1417
    hui_test_dataset = load_hui_dataset(args, hui_config, split="val", num_workers=0) # expected 407
    ssup_train_dataset = load_hui_dataset(args, ssup_config, split="train", num_workers=0) # expected 6098
    ssup_test_dataset = load_hui_dataset(args, ssup_config, split="val", num_workers=0) # expected 4875

    print("hui_train_dataset length: ", len(hui_train_dataset))
    print("hui_test_dataset length: ", len(hui_test_dataset))
    print("ssup_train_dataset length: ", len(ssup_train_dataset))
    print("ssup_test_dataset length: ", len(ssup_test_dataset))

    os.makedirs(args.output_dir, exist_ok=True)

    exports = [
        ("hui_train_data.pkl", hui_train_dataset, hui_config),
        ("hui_test_dataset.pkl", hui_test_dataset, hui_config),
        ("ssup_train_dataset.pkl", ssup_train_dataset, ssup_config),
        ("ssup_test_dataset.pkl", ssup_test_dataset, ssup_config),
    ]

    for filename, dataset, config in exports:
        output_path = os.path.join(args.output_dir, filename)
        print(f"Extracting {filename} ({len(dataset)} samples)...")
        payload = extract_dataset_to_dict(dataset, config)
        assert payload["data"].shape == (len(dataset), 48, 17, 3), payload["data"].shape
        assert payload["labels"].shape == (len(dataset),), payload["labels"].shape
        assert payload["ids"].shape == (len(dataset),), payload["ids"].shape
        joblib.dump(payload, output_path)
        print(f"Saved {output_path} — data {payload['data'].shape}, labels {payload['labels'].shape}, ids {payload['ids'].shape}")

