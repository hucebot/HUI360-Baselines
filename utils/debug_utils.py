## Plotting function for debug...
import os
import sys
import time
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join("..", here))

import numpy as np
import torch
import matplotlib.pyplot as plt


from utils.print_utils import prInfo, prWarning, prError, prSuccess, prDebug
from utils.visualize_utils import COCO18_SKELETON_CONNECTIONS, COCO18_SKELETON_COLORS


def export_unique_track_identifiers(dataset, split: str, cross_eval_type: str, output_dir: str = None, add_label: bool = False):
    """
    Export unique track identifiers from a dataset to a text file.
    
    Args:
        dataset: The HUIInteract360 or HUIInteract360Light dataset instance.
        split: The split name (e.g., 'train' or 'test').
        cross_eval_type: The cross evaluation type (e.g., 'hui_train_vs_ssup_test').
        output_dir: Optional output directory. If None, uses 'datasets/tracks_saved_identifiers/'.
        add_label: If True, save each track as "UNIQUEID, LABEL" where LABEL is 0 or 1.
    
    Returns:
        str: Path to the saved file.
    """
    # Get unique track identifiers from the dataset
    # The dataset has idx_to_unique_track_identifier which maps indices to track identifiers
    unique_track_identifiers = sorted(set(dataset.idx_to_unique_track_identifier))
    
    # If add_label is True, build a mapping from unique track identifier to label
    track_to_label = {}
    if add_label:
        for idx, track_id in enumerate(dataset.idx_to_unique_track_identifier):
            if track_id not in track_to_label:
                # Get label for this track from the dataset's label mapping
                label = int(dataset.idx_to_label[idx])
                track_to_label[track_id] = label
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(here, "..", "datasets", "tracks_saved_identifiers")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        prInfo(f"Created output directory: {output_dir}")
    
    # Generate timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{split}_{cross_eval_type}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Write unique track identifiers to file
    with open(filepath, "w") as f:
        for track_id in unique_track_identifiers:
            if add_label:
                label = track_to_label[track_id]
                f.write(f"{track_id}, {label}\n")
            else:
                f.write(f"{track_id}\n")
    
    prSuccess(f"Exported {len(unique_track_identifiers)} unique track identifiers to {filepath}")
    
    return filepath

def plot_input_tensor(input_tensor, metadata, labels, predictions):
    ### for debugging purposes
    input_tensor = input_tensor.detach().cpu().numpy()
    B,C,T,V = input_tensor.shape
    print(f"input_tensor shape : B = {B}, C = {C}, T = {T}, V = {V}")
    # plt.subplots(B,T, figsize=(10, 20)) # B rows, T columns
    # for b in range(B):
    #     for t in range(T):
    #         plt.subplot(B,T,b*T+t+1)
    #         plt.scatter(input_tensor[b,0,t,:], input_tensor[b,1,t,:], c=input_tensor[b,2,t,:], cmap='viridis')
    #         plt.title(f"Frame {t}")
    #         plt.colorbar()
    
    B = min(B,8)
    
    if B == 1:
        
        plt.figure(figsize=(10, 10))
        for t in range(T):
            if t == 0 or t == T-1:
                plt.scatter(input_tensor[0,0,t,:], input_tensor[0,1,t,:], c=input_tensor[0,2,t,:], cmap='viridis', alpha=t/T)
            for connection, color in zip(COCO18_SKELETON_CONNECTIONS, COCO18_SKELETON_COLORS):
                start_idx, end_idx = connection
                color = (color[0]/255, color[1]/255, color[2]/255)
                plt.plot([input_tensor[0,0,t,start_idx], input_tensor[0,0,t,end_idx]], [input_tensor[0,1,t,start_idx], input_tensor[0,1,t,end_idx]], color=color, alpha=t/T)
            
        plt.xlim(0, 1)
        plt.ylim(1, 0)
        plt.title("Label: {:.0f}, NLL: {:.2f}".format(labels, predictions[0]))
                
    else:
        
        plt.subplots(4,B//4, figsize=(20, 20)) # 4 rows, B//4 columns
        for b in range(B):
            plt.subplot(4,B//4,b+1)
            for t in range(T):
                if t == 0 or t == T-1:
                    plt.scatter(input_tensor[b,0,t,:], input_tensor[b,1,t,:], c=input_tensor[b,2,t,:], cmap='viridis', alpha=t/T)
                for connection, color in zip(COCO18_SKELETON_CONNECTIONS, COCO18_SKELETON_COLORS):
                    start_idx, end_idx = connection
                    color = (color[0]/255, color[1]/255, color[2]/255)
                    plt.plot([input_tensor[b,0,t,start_idx], input_tensor[b,0,t,end_idx]], [input_tensor[b,1,t,start_idx], input_tensor[b,1,t,end_idx]], color=color, alpha=t/T)
                
            plt.xlim(0, 1)
            plt.ylim(1, 0)
            plt.title("Label: {:.0f}, NLL: {:.2f}".format(labels[b].item(), predictions[b].item()))
                
    plt.show()
    
    
def plot_input_tensors_skformer(coco_tensor, sapiens_tensor, input_tensor, input_format_name):
    B,C,T,V,M = input_tensor.shape
    print(f"input_tensor shape : B = {B}, C = {C}, T = {T}, V = {V}, M = {M}")
    if coco_tensor is not None:
        Bcoco,Tcoco,Vcoco,Ccoco = coco_tensor.shape
        assert Bcoco == B, "coco_tensor and input_tensor must have the same batch size"
        assert Tcoco == T, "coco_tensor and input_tensor must have the same time steps"
        assert Ccoco == C, "coco_tensor and input_tensor must have the same number of channels"
    if sapiens_tensor is not None:
        Bsapiens,Tsapiens,Vsapiens,Csapiens = sapiens_tensor.shape
        assert Bsapiens == B, "sapiens_tensor and input_tensor must have the same batch size"
        assert Tsapiens == T, "sapiens_tensor and input_tensor must have the same time steps"
        assert Csapiens == C, "sapiens_tensor and input_tensor must have the same number of channels"
    ### FOR DEBUG
    ntu_nospine_mid_joint_names = [
        "base of the spine",
        "neck",
        "head",
        "left shoulder",
        "left elbow",
        "left wrist",
        "left hand",
        "right shoulder",
        "right elbow",
        "right wrist",
        "right hand",
        "left hip",
        "left knee",
        "left ankle",
        "left foot",
        "right hip",
        "right knee",
        "right ankle",
        "right foot",
        "spine",
        "tip of the left hand",
        "left thumb",
        "tip of the right hand",
        "right thumb"
    ]
    
    ntu_nospine_mid_joint_links = [
        # Spine
        (0, 19),   # base of spine -> spine
        (19, 1),   # spine -> neck
        (1, 2),    # neck -> head

        # Left arm
        (1, 3),    # neck -> left shoulder
        (3, 4),    # left shoulder -> left elbow
        (4, 5),    # left elbow -> left wrist
        (5, 6),    # left wrist -> left hand
        (6, 20),   # left hand -> tip of the left hand
        (5, 21),   # left wrist -> left thumb

        # Right arm
        (1, 7),    # neck -> right shoulder
        (7, 8),    # right shoulder -> right elbow
        (8, 9),    # right elbow -> right wrist
        (9, 10),   # right wrist -> right hand
        (10, 22),  # right hand -> tip of the right hand
        (9, 23),   # right wrist -> right thumb

        # Left leg
        (0, 11),   # base of spine -> left hip
        (11, 12),  # left hip -> left knee
        (12, 13),  # left knee -> left ankle
        (13, 14),  # left ankle -> left foot

        # Right leg
        (0, 15),   # base of spine -> right hip
        (15, 16),  # right hip -> right knee
        (16, 17),  # right knee -> right ankle
        (17, 18),  # right ankle -> right foot
    ]
    
    nwucla_joint_names = [
        "base spine",
        "middle spine",
        "neck",
        "head",
        "left shoulder",
        "left elbow",
        "left wrist",
        "left hand",
        "right shoulder",
        "right elbow",
        "right wrist",
        "right hand",
        "left hip",
        "left knee",
        "left ankle",
        "left foot",
        "right hip",
        "right knee",
        "right ankle",
        "right foot"]
    
    nwucla_joint_links = [
        # Spine chain
        (0, 1),   # base spine -> middle spine
        (1, 2),   # middle spine -> neck
        (2, 3),   # neck -> head
        
        # Left arm
        (2, 4),   # neck -> left shoulder
        (4, 5),   # left shoulder -> left elbow
        (5, 6),   # left elbow -> left wrist
        (6, 7),   # left wrist -> left hand
        
        # Right arm
        (2, 8),   # neck -> right shoulder
        (8, 9),   # right shoulder -> right elbow
        (9, 10),  # right elbow -> right wrist
        (10, 11), # right wrist -> right hand
        
        # Left leg
        (0, 12),  # base spine -> left hip
        (12, 13), # left hip -> left knee
        (13, 14), # left knee -> left ankle
        (14, 15), # left ankle -> left foot
        
        # Right leg
        (0, 16),  # base spine -> right hip
        (16, 17), # right hip -> right knee
        (17, 18), # right knee -> right ankle
        (18, 19), # right ankle -> right foot
    ]
    
    # COCO/ViTPose 17 keypoints
    coco_joint_names = [
        "nose",          # 0
        "left_eye",      # 1
        "right_eye",     # 2
        "left_ear",      # 3
        "right_ear",     # 4
        "left_shoulder", # 5
        "right_shoulder",# 6
        "left_elbow",    # 7
        "right_elbow",   # 8
        "left_wrist",    # 9
        "right_wrist",   # 10
        "left_hip",      # 11
        "right_hip",     # 12
        "left_knee",     # 13
        "right_knee",    # 14
        "left_ankle",    # 15
        "right_ankle"    # 16
    ]
    
    coco_joint_links = [
        # Face
        (0, 1),   # nose -> left_eye
        (0, 2),   # nose -> right_eye
        (1, 3),   # left_eye -> left_ear
        (2, 4),   # right_eye -> right_ear
        
        # Shoulders
        (5, 6),   # left_shoulder -> right_shoulder
        
        # Left arm
        (5, 7),   # left_shoulder -> left_elbow
        (7, 9),   # left_elbow -> left_wrist
        
        # Right arm
        (6, 8),   # right_shoulder -> right_elbow
        (8, 10),  # right_elbow -> right_wrist
        
        # Torso
        (5, 11),  # left_shoulder -> left_hip
        (6, 12),  # right_shoulder -> right_hip
        (11, 12), # left_hip -> right_hip
        
        # Left leg
        (11, 13), # left_hip -> left_knee
        (13, 15), # left_knee -> left_ankle
        
        # Right leg
        (12, 14), # right_hip -> right_knee
        (14, 16), # right_knee -> right_ankle
    ]
    
    # Sapiens (without face) 63 keypoints
    sapiens_joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",           # 0-4
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",     # 5-8
        "left_hip", "right_hip", "left_knee", "right_knee",                 # 9-12
        "left_ankle", "right_ankle",                                         # 13-14
        "left_big_toe", "left_small_toe", "left_heel",                       # 15-17
        "right_big_toe", "right_small_toe", "right_heel",                    # 18-20
        "right_thumb4", "right_thumb3", "right_thumb2", "right_thumb_third_joint",  # 21-24
        "right_forefinger4", "right_forefinger3", "right_forefinger2", "right_forefinger_third_joint",  # 25-28
        "right_middle_finger4", "right_middle_finger3", "right_middle_finger2", "right_middle_finger_third_joint",  # 29-32
        "right_ring_finger4", "right_ring_finger3", "right_ring_finger2", "right_ring_finger_third_joint",  # 33-36
        "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2", "right_pinky_finger_third_joint",  # 37-40
        "right_wrist",  # 41
        "left_thumb4", "left_thumb3", "left_thumb2", "left_thumb_third_joint",  # 42-45
        "left_forefinger4", "left_forefinger3", "left_forefinger2", "left_forefinger_third_joint",  # 46-49
        "left_middle_finger4", "left_middle_finger3", "left_middle_finger2", "left_middle_finger_third_joint",  # 50-53
        "left_ring_finger4", "left_ring_finger3", "left_ring_finger2", "left_ring_finger_third_joint",  # 54-57
        "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2", "left_pinky_finger_third_joint",  # 58-61
        "left_wrist"   # 62
    ]
    
    sapiens_joint_links = [
        # Legs
        (13, 11), (11, 9), (14, 12), (12, 10), (9, 10),
        # Torso
        (5, 9), (6, 10), (5, 6), (5, 7), (6, 8),
        # Arms to wrists
        (7, 62), (8, 41),
        # Face
        (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
        # Feet
        (13, 15), (13, 16), (13, 17), (14, 18), (14, 19), (14, 20),
        # Left hand fingers
        (62, 45), (45, 44), (44, 43), (43, 42),  # thumb
        (62, 49), (49, 48), (48, 47), (47, 46),  # forefinger
        (62, 53), (53, 52), (52, 51), (51, 50),  # middle
        (62, 57), (57, 56), (56, 55), (55, 54),  # ring
        (62, 61), (61, 60), (60, 59), (59, 58),  # pinky
        # Right hand fingers
        (41, 24), (24, 23), (23, 22), (22, 21),  # thumb
        (41, 28), (28, 27), (27, 26), (26, 25),  # forefinger
        (41, 32), (32, 31), (31, 30), (30, 29),  # middle
        (41, 36), (36, 35), (35, 34), (34, 33),  # ring
        (41, 40), (40, 39), (39, 38), (38, 37),  # pinky
    ]
    
    # INSERT_YOUR_CODE
    import matplotlib.pyplot as plt

    # Get the first sample (batch 0) and time indices: 0, T//2, T-1
    times_to_plot = [0, T // 2, T - 1]
    
    # Determine number of rows: 1 for input + optional COCO + optional Sapiens
    num_rows = 1 + (1 if coco_tensor is not None else 0) + (1 if sapiens_tensor is not None else 0)
    
    fig, axs = plt.subplots(num_rows, len(times_to_plot), figsize=(5 * len(times_to_plot), 5 * num_rows))
    if len(times_to_plot) == 1:
        axs = axs.reshape(num_rows, 1)
    if num_rows == 1:
        axs = axs.reshape(1, len(times_to_plot))
    
    # Select joint names and links based on input format
    if input_format_name == "nwucla":
        input_joint_names = nwucla_joint_names
        input_joint_links = nwucla_joint_links
    elif input_format_name == "ntu_nospine_mid":
        input_joint_names = ntu_nospine_mid_joint_names
        input_joint_links = ntu_nospine_mid_joint_links
    else:
        raise ValueError(f"Unknown input format: {input_format_name}")
    
    # input_tensor shape: B, C, T, V, M with M=1
    # coco_tensor shape: B, T, V, C (if not None)
    # sapiens_tensor shape: B, T, V, C with V=63 (if not None)
    for idx, t in enumerate(times_to_plot):
        row_idx = 0
        
        # --- Input tensor skeleton (first row) ---
        joints = input_tensor[0, :2, t, :, 0].detach().cpu().numpy()  # shape: (2, V)
        x, y = joints[0], joints[1]
        ax = axs[row_idx, idx]
        # Draw skeleton links
        for j1, j2 in input_joint_links:
            ax.plot([x[j1], x[j2]], [y[j1], y[j2]], 'b-', linewidth=1.5, alpha=0.7)
        # Draw joints
        ax.scatter(x, y, c='b', s=30, zorder=5)
        for i, name in enumerate(input_joint_names):
            ax.text(x[i], y[i], name, fontsize=7)
        ax.set_title(f"Input ({input_format_name}) at t={t}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.invert_yaxis()  # Image coordinates: y increases downward
        ax.axis('equal')
        row_idx += 1
        
        # --- COCO skeleton (if available) ---
        if coco_tensor is not None:
            coco_joints = coco_tensor[0, t, :, :2].detach().cpu().numpy()  # shape: (17, 2)
            cx, cy = coco_joints[:, 0], coco_joints[:, 1]
            ax_coco = axs[row_idx, idx]
            # Draw skeleton links
            for j1, j2 in coco_joint_links:
                ax_coco.plot([cx[j1], cx[j2]], [cy[j1], cy[j2]], 'r-', linewidth=1.5, alpha=0.7)
            # Draw joints
            ax_coco.scatter(cx, cy, c='r', s=30, zorder=5)
            for i, name in enumerate(coco_joint_names):
                ax_coco.text(cx[i], cy[i], name, fontsize=7)
            ax_coco.set_title(f"COCO/ViTPose at t={t}")
            ax_coco.set_xlabel("x")
            ax_coco.set_ylabel("y")
            ax_coco.invert_yaxis()  # Image coordinates: y increases downward
            ax_coco.axis('equal')
            row_idx += 1
        
        # --- Sapiens skeleton (if available) ---
        if sapiens_tensor is not None:
            sapiens_joints = sapiens_tensor[0, t, :, :2].detach().cpu().numpy()  # shape: (63, 2)
            sx, sy = sapiens_joints[:, 0], sapiens_joints[:, 1]
            ax_sap = axs[row_idx, idx]
            # Draw skeleton links
            for j1, j2 in sapiens_joint_links:
                ax_sap.plot([sx[j1], sx[j2]], [sy[j1], sy[j2]], 'g-', linewidth=1.5, alpha=0.7)
            # Draw joints
            ax_sap.scatter(sx, sy, c='g', s=20, zorder=5)
            # Only label a subset to avoid clutter
            for i in range(len(sapiens_joint_names)):
                ax_sap.text(sx[i], sy[i], sapiens_joint_names[i], fontsize=6)
            ax_sap.set_title(f"Sapiens (no face) at t={t}")
            ax_sap.set_xlabel("x")
            ax_sap.set_ylabel("y")
            ax_sap.invert_yaxis()  # Image coordinates: y increases downward
            ax_sap.axis('equal')
        
    plt.tight_layout()
    plt.show()
    
def update_old_config_dict(config):
    if "remove_joints" not in config:
        prWarning("remove_joints not in config, setting to None")
        config["remove_joints"] = None
    if "perspective_reprojection" not in config:
        prWarning("perspective_reprojection not in config, setting to None")
        config["perspective_reprojection"] = {
            "do_perspective_reprojection": False,
            "hcenter": 0.0,
            "vcenter": 0.0,
            "hfov": 94.0,
            "vfov": 68.0,
            "target_width": 1280
        }
    if "format_by_channel" not in config:
        prWarning("format_by_channel not in config, setting to False")
        config["format_by_channel"] = False
    if "mb_desired_return" not in config:
        prWarning("mb_desired_return not in config, setting to representation")
        config["mb_desired_return"] = "representation"
    if "skateformer_in_channels" not in config:
        prWarning("skateformer_in_channels not in config, setting to 3")
        config["skateformer_in_channels"] = 3
    if "stgcn_in_channels" not in config:
        prWarning("stgcn_in_channels not in config, setting to 3")
        config["stgcn_in_channels"] = 3
    if "stgcn_edge_importance_weighting" not in config:
        prWarning("stgcn_edge_importance_weighting not in config, setting to True")
        config["stgcn_edge_importance_weighting"] = True
    if "stgcn_layout" not in config:
        prWarning("stgcn_layout not in config, setting to openpose")
        config["stgcn_layout"] = "openpose"
    if "cutoffs_filtering" not in config:
        prWarning("cutoffs_filtering not in config, setting to True")
        config["cutoffs_filtering"] = True
    if "use_polar_coordinates" not in config:
        prWarning("use_polar_coordinates not in config, setting to False")
        config["use_polar_coordinates"] = False
    if "mb_input_norm" not in config:
        prWarning("mb_input_norm not in config, setting to scale")
        config["mb_input_norm"] = "scale"
    if "mb_head_dropout" not in config:
        prWarning("mb_head_dropout not in config, setting to 0.5")
        config["mb_head_dropout"] = 0.5
    if "mb_head_hidden_dim" not in config:
        prWarning("mb_head_hidden_dim not in config, setting to 2048")
        config["mb_head_hidden_dim"] = 2048
    if "lr_scheduler_type" not in config:
        prWarning("lr_scheduler_type not in config, setting to None")
        config["lr_scheduler_type"] = "ExponentialDecay"
    if "do_recenter_interaction_zone" not in config:
        prWarning("do_recenter_interaction_zone not in config, setting to False")
        config["do_recenter_interaction_zone"] = False
    if "random_flip_horizontal_train" not in config:
        prWarning("random_flip_horizontal_train not in config, setting to False")
        config["random_flip_horizontal_train"] = False
    if "random_flip_horizontal_val" not in config:
        prWarning("random_flip_horizontal_val not in config, setting to False")
        config["random_flip_horizontal_val"] = False
    if "random_jitter_position_train" not in config:
        prWarning("random_jitter_position_train not in config, setting to (0.0, 0.0)")
        config["random_jitter_position_train"] = (0.0, 0.0)
    if "random_jitter_position_val" not in config:
        prWarning("random_jitter_position_val not in config, setting to (0.0, 0.0)")
        config["random_jitter_position_val"] = (0.0, 0.0)
    if "center_on_onset_train" not in config:
        prWarning("center_on_onset_train not in config, setting to False")
        config["center_on_onset_train"] = False
    if "center_on_onset_val" not in config:
        prWarning("center_on_onset_val not in config, setting to False")
        config["center_on_onset_val"] = False
    if "do_recentering_train" not in config:
        prWarning("do_recentering_train not in config, setting to False")
        config["do_recentering_train"] = False
    if "do_recentering_val" not in config:
        prWarning("do_recentering_val not in config, setting to False")
        config["do_recentering_val"] = False
    if "do_fix_keypoints_outside_box_train" not in config:
        prWarning("do_fix_keypoints_outside_box_train not in config, setting to True")
        config["do_fix_keypoints_outside_box_train"] = True
    if "do_fix_keypoints_outside_box_val" not in config:
        prWarning("do_fix_keypoints_outside_box_val not in config, setting to True")
        config["do_fix_keypoints_outside_box_val"] = True
    if "normalize_keypoints_in_track" not in config:
        prWarning("normalize_keypoints_in_track not in config, setting to none")
        config["normalize_keypoints_in_track"] = "none"
    if "inputs_per_track_stride_train" not in config:
        prWarning("inputs_per_track_stride_train not in config, setting to -1")
        config["inputs_per_track_stride_train"] = -1
    if "inputs_per_track_stride_val" not in config:
        prWarning("inputs_per_track_stride_val not in config, setting to -1")
        config["inputs_per_track_stride_val"] = -1
    if "stg_nf_hidden_channels" not in config:
        prWarning("stg_nf_hidden_channels not in config, setting to 0")
        config["stg_nf_hidden_channels"] = 0
    if "hf_dataset_revision" not in config:
        prWarning("hf_dataset_revision not in config, setting to 3c8a342548534b6b92d32b0099e266962facdf45")
        config["hf_dataset_revision"] = "3c8a342548534b6b92d32b0099e266962facdf45"
        
    return config