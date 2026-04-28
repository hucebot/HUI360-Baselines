# Import standard libraries
import argparse
from math import e
import os
# Define the here variable to be the directory of the current file
here = os.path.dirname(os.path.abspath(__file__))
from tqdm import tqdm
import joblib
from collections import OrderedDict
import socket
hostname = socket.gethostname()
import warnings
warnings.filterwarnings("ignore")

# Import torch libraries
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

# Import random library
import random
import numpy as np

# import metrics and classifiers from scikit-learn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, average_precision_score

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

# Import the predictors classes
from predictors.mlp import MLPInteractionPredictor
from predictors.lstm import LSTMInteractionPredictor
from predictors.STG_NF.model_pose import STG_NF
from predictors.STGCN.net.st_gcn import Model as STGCN
from predictors.SkateFormer.model.SkateFormer import SkateFormer

# Import custom utils
from utils.print_utils import *
from utils.loader_utils import load_hui_dataset
from utils.data_utils import VITPOSE_KEYPOINTS_NAMES
from utils.other_utils import read_yaml_to_dic
from utils.debug_utils import plot_input_tensor, update_old_config_dict, plot_input_tensors_skformer, export_unique_track_identifiers
from utils.eval_utils import get_best_threshold_f1


def evaluate(model, dataloader, device, config=None):
    """Evaluate the model on a dataset."""
    model.eval()
    all_labels = []
    all_probabilities = []
    all_unique_track_identifiers = []
    
    total_time = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch_idx, (input_tensor, label, metadata, images_tensor, masks_tensor) in enumerate(progress_bar):
            
            # Store the identifiers of the samples in the batch
            for track_identifier in metadata["unique_track_identifier"]:
                all_unique_track_identifiers.append(track_identifier)
                
            start.record()
            
            B,T,D = input_tensor.shape

            model_kwargs = {}
            if config["use_polar_coordinates"] == True:
                # just make sure of basic compatibility things
                assert(type(model) == LSTMInteractionPredictor), "Polar coordinates are only compatible with LSTMInteractionPredictor"
                assert(config["normalize_keypoints_in_track"] == "none"), "Polar coordinates are only compatible with normalize_keypoints_in_track = 'none'"
                assert(config["normalize_keypoints_in_box"] == False), "Polar coordinates are only compatible with normalize_keypoints_in_box = False"
                assert(config["normalize_in_image"] == True), "Polar coordinates are only compatible with normalize_in_image = True"
                assert(config["standardize_data"] == "mask_only"), "Polar coordinates are only compatible with standardize_data = 'mask_only'"
                
                # add the polar coordinates to the input tensor
                input_tensor, data_columns = convert_to_polar_coordinates(input_tensor, dataloader.dataset.data_columns_in_dataset)
                
            if type(model) == STGCN:
                in_channels = model.in_channels
                input_tensor = input_tensor_to_format_by_channel(input_tensor, metadata, dataloader.dataset.data_columns_in_dataset) # B,T,V,C
                input_tensor = keypoints17_to_coco18_torch(input_tensor) # B,T,V,C with C=18
                input_tensor = input_tensor.permute(0, 3, 1, 2).contiguous() # B,C,T,V with C=18
                input_tensor = input_tensor.unsqueeze(-1) # B,C,T,V,M with M=1
                input_tensor = input_tensor[:,:in_channels,:,:,:] # B,C,T,V,M with C=in_channels
                # print(f"input_tensor shape: {input_tensor.shape}")
                # B, C, T, V, M with M=1 

            elif type(model) == SkateFormer:
                # D = 56 for ViTPose, 245 for ViTPose + Sapiens, 194 for Sapiens
                # + bbox + masksize in every case

                if D == 56:
                    coco_tensor = input_tensor_to_format_by_channel(input_tensor.clone(), metadata, dataloader.dataset.data_columns_in_dataset) # B,T,V,C with V=17
                    sapiens_tensor = None
                    input_format_name = "nwucla"
                    input_tensor = coco2nwucla(coco_tensor) # B,T,20,3
                elif D == 245:
                    coco_tensor = input_tensor_to_format_by_channel(input_tensor.clone(), metadata, dataloader.dataset.data_columns_in_dataset) # B,T,V,C with V=17
                    sapiens_tensor = input_tensor_to_format_by_channel_sapiens_without_face(input_tensor.clone(), metadata, dataloader.dataset.data_columns_in_dataset) # B,T,V,C with V=63
                    input_format_name = "ntu_nospine_mid"
                    input_tensor = sapiensnoface2nturgbd_nospine_mid(sapiens_tensor) # B,T,24,3
                elif D == 194:
                    coco_tensor = None
                    sapiens_tensor = input_tensor_to_format_by_channel_sapiens_without_face(input_tensor.clone(), metadata, dataloader.dataset.data_columns_in_dataset) # B,T,V,C with V=63
                    input_format_name = "ntu_nospine_mid"
                    input_tensor = sapiensnoface2nturgbd_nospine_mid(sapiens_tensor) # B,T,24,3
                else:
                    raise ValueError(f"Invalid number of input columns: {D} (expect 56, 245 or 194)")
                    
                input_tensor = input_tensor.permute(0, 3, 1, 2).contiguous() # B,C,T,V
                input_tensor = input_tensor.unsqueeze(-1) # B,C,T,V,M with M=1
                
             
                index_t = (2 * (torch.arange(0, T, device=device) / (T-1))) - 1 # in [-1, 1]
                
                model_kwargs["index_t"] = index_t
            
            elif type(model) == STG_NF:
                # rearange by channels (_x, _y, _score)
                input_tensor = input_tensor_to_format_by_channel(input_tensor, metadata, dataloader.dataset.data_columns_in_dataset) # B,T,V,C with V=17            
                input_tensor = keypoints17_to_coco18_torch(input_tensor) # to Openpose format
                input_tensor = input_tensor.permute(0, 3, 1, 2).contiguous() # B,C,T,V with C=18
                # input_tensor = torch.tensor(input_tensor).to(device)
                # B,C,T,V = input_tensor.shape # with B=batch, C=channels=3, T=time, V=keypoints=18
                
                score = torch.ones((B,T)).to(device).amin(dim=-1) # same score for all samples (originally this is a score from the input)
                label_model = torch.ones_like(label).to(device) # at eval time only use ones_like, label_model = torch.where(label_model == 0, 1, -1)

                input_tensor = input_tensor[:, :2].float()
                model_kwargs["score"] = score
                model_kwargs["label"] = label_model
                            
            # Move to device
            input_tensor = input_tensor.to(device)
            label = label.to(device).float()

            # Forward pass
            model_output = model(input_tensor, **model_kwargs) # Note that in the case of STG_NF, the return is actually the nll itself
            
            probabilities = torch.sigmoid(model_output.squeeze()) if type(model) != STG_NF else model_output # torch.tensor of shape (B,)
            # make a 1D tensor if probabilities is a 0D tensor (case of B=1)
            if probabilities.ndim == 0:
                probabilities = probabilities.unsqueeze(0) # torch.tensor of shape (1,)
            
            all_labels.extend(label.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            end.record()
            torch.cuda.synchronize()
            iter_time = start.elapsed_time(end)
            total_time += iter_time
            # Update progress bar
            progress_bar.set_postfix({
                'Time': f'{iter_time:.2f} ms',
                'Avg Time': f'{total_time/(batch_idx+1):.2f} ms'
            })
    
    return all_labels, all_probabilities, all_unique_track_identifiers

def main(args, model_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Make sure the checkpoint contains the model weights
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict' key. Cannot load model weights.")
    
    # Extract config and handle backward compatibility
    if 'config' not in checkpoint:
        if 'hyperparameters' in checkpoint:
            checkpoint["config"] = checkpoint["hyperparameters"]
        prError("Checkpoint does not contain 'config' or 'hyperparameters' key. Cannot recreate dataloader.")
        exit(1)

    config = checkpoint['config']
    prInfo("Loaded config from checkpoint")
    config = update_old_config_dict(config)
    
    # Print some info
    prInfo(f"Model type: {config['force_model_type']}")
    prInfo(f"Cross evaluation type: {config['cross_eval_type']}")

    # Print checkpoint infos
    assert(key in checkpoint for key in ['val_auc', 'val_f1', 'val_f1_with_adaptative_threshold', 'val_ap', 'epoch']), f"Checkpoint does not contain expected keys: {keys}"
    prInfo(f"Checkpoint validation AUC (expected): {checkpoint['val_auc']:.4f}")
    prInfo(f"Checkpoint validation AP (expected): {checkpoint['val_ap']:.4f}")
    prInfo(f"Checkpoint validation F1 with adaptative threshold (expected): {checkpoint['val_f1_with_adaptative_threshold']:.4f}")
    prInfo(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    
    # Create validation dataset
    if config["val_tracks_filename"] != "all":
        tracks_file_path = os.path.join(here, "datasets", "tracks_saved_identifiers", config["val_tracks_filename"])
        if not os.path.exists(tracks_file_path):
            prError(f"Validation tracks file not found: {tracks_file_path}")
            exit(1)
        val_tracks = [line.strip() for line in open(tracks_file_path)]
    else:
        val_tracks = "all"

    val_dataset = load_hui_dataset(args, config, split="val", num_workers=0)

    prInfo(f"Validation positives: {val_dataset.total_positives_tracks}")
    prInfo(f"Validation negatives: {val_dataset.total_negatives_tracks}")
    prInfo(f"Validation samples: {len(val_dataset)}")

    # Get input dimensions and sequence length
    input_dim = len(val_dataset.data_columns_in_dataset)
    if config["use_polar_coordinates"] == True:
        prWarning("Using polar coordinates adding 2x the number of input dimensions for the x,y coordinates")
        for col in val_dataset.data_columns_in_dataset:
            if col.endswith("_x") or col.endswith("_y"):
                # use sin/cos embedding for each of the x and y coordinates so it doubles their number (but score, and non x,y coordinates remains the same)
                input_dim += 1
            if col == "xmin" or col == "xmax" or col == "ymin" or col == "ymax":
                # use sin/cos embedding for each of the x and y coordinates so it doubles their number (but score, and non x,y coordinates remains the same)
                input_dim += 1
    
    sequence_length = config["input_length_in_frames"] // config["subsample_frames"]
    
    prInfo(f"Input dimension: {input_dim}")
    prInfo(f"Sequence length: {sequence_length} (input_length_in_frames = {config['input_length_in_frames']}, subsample_frames = {config['subsample_frames']})")

    # Create data loader
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Instantiate model
    if config["force_model_type"] == "mlp":
        model = MLPInteractionPredictor(
            input_dim=input_dim,
            sequence_length=sequence_length,
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"]
        ).to(device)
        
    elif config["force_model_type"] == "lstm":
        model = LSTMInteractionPredictor(
            input_dim=input_dim,
            sequence_length=sequence_length,
            hidden_dim=config["lstm_hidden_dim"],
            num_layers=config["lstm_num_layers"],
            dropout=config["lstm_dropout"],
            bidirectional=False
        ).to(device)
        

    elif config["force_model_type"] == "stg_nf":
        model = STG_NF(device=device,
                        pose_shape=(2, sequence_length, 18),
                        hidden_channels=config["stg_nf_hidden_channels"],
                        K=config["stg_nf_K"],
                        L=config["stg_nf_L"],
                        R=config["stg_nf_R"],
                        actnorm_scale=config["stg_nf_actnorm_scale"],
                        flow_permutation="permute",
                        flow_coupling="affine",
                        LU_decomposed=True,
                        learn_top=False,
                        edge_importance=config["stg_nf_edge_importance"],
                        temporal_kernel_size=None,
                        strategy="uniform",
                        max_hops=config["stg_nf_max_hops"],).to(device)

    elif config["force_model_type"] == "stgcn":
        model = STGCN(
            in_channels=config["stgcn_in_channels"],
            num_class=1,
            graph_args={"layout": config["stgcn_layout"], "strategy": 'spatial'},
            edge_importance_weighting=config["stgcn_edge_importance_weighting"],
        ).to(device)
        
    elif config["force_model_type"] == "skateformer":
        assert(sequence_length % 8 == 0), "Sequence length must be divisible by 8 for SkateFormer"
        Tdim = sequence_length // 8
        ncolumns_input = len(config["include_columns"]) 
        # 74 = D3 (ViTPose) will be mapped to NW-UCLA (20 joints) (in this case at loading, D=56)
        # 263 = D8 (ViTPose + Sapiens) will be mapped to NTU without spine (24 joints) (in this case at loading, D=245)
        # 212 = D9 (Sapiens) will be mapped to NTU without spine (24 joints) (in this case at loading, D=194)
        if ncolumns_input == 74:
            num_joints_mapped = 20
            types_spatial_sizes = [(Tdim, 4), (Tdim, 5), (Tdim, 4), (Tdim, 5)]
        elif ncolumns_input == 263 or ncolumns_input == 212:
            num_joints_mapped = 24
            types_spatial_sizes = [(Tdim, 8), (Tdim, 12), (Tdim, 8), (Tdim, 12)]
        else:
            raise ValueError(f"Invalid number of input columns: {ncolumns_input} (expect 74, 263 or 212 i.e. D3, D8 or D9)")

        model = SkateFormer(
            in_channels=config["skateformer_in_channels"],
            depths=(2, 2, 2, 2),
            channels=(96, 192, 192, 192),
            num_classes=1,
            embed_dim=96,
            num_people=1,
            num_points=num_joints_mapped,
            kernel_size=7,
            num_heads=32,
            attn_drop=0.5,
            head_drop=0.0,
            rel=True,
            drop_path=0.2,
            type_1_size=types_spatial_sizes[0],
            type_2_size=types_spatial_sizes[1],
            type_3_size=types_spatial_sizes[2],
            type_4_size=types_spatial_sizes[3],
            mlp_ratio=1.0,
            index_t=True,
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    prSuccess("Model weights loaded successfully")
    
    if isinstance(model, STG_NF):
        model.set_actnorm_init()
        prSuccess("ActNorm2d initialized for STG_NF")

    # Run evaluation
    prInfo("Starting evaluation...")
    all_labels, all_probabilities, all_unique_track_identifiers = evaluate(model, val_dataloader, device, config=config)

    # Compute metrics
    auc = roc_auc_score(all_labels, all_probabilities)
    best_threshold, best_f1 = get_best_threshold_f1(all_labels, all_probabilities, thresholds_min_max=True)
    accuracy_at_best_threshold = np.average(all_labels == (all_probabilities >= best_threshold))
    ap = average_precision_score(all_labels, all_probabilities)

    # Print results
    prSuccess("\n" + "="*50)
    prSuccess("EVALUATION RESULTS")
    prSuccess("="*50)
    prSuccess(f"AUC:       {auc:.4f}")
    prSuccess(f"AP:        {ap:.4f}")
    prSuccess(f"F1@{best_threshold:.3f}:  {best_f1:.4f}")
    prSuccess(f"Acc@{best_threshold:.3f}: {accuracy_at_best_threshold:.4f}")
    prSuccess("="*50)

    return all_labels, all_probabilities, all_unique_track_identifiers
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on a trained model')
    parser.add_argument("--model_path", "-m", required=True, type=str, help="Path to the .pth model checkpoint file. Required.")
    parser.add_argument("--num_workers", default=None, type=int, help="Number of workers for dataloader (default: auto)")
    parser.add_argument("--hf_local_dir", "-hld", default="default", type=str, help="HF local directory (default: default, ie ./datasets/hf_data)")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Verbose mode")

    args = parser.parse_args()
    
    # For compatibility with load_hui_dataset
    args.preload_data = False
    args.preload_only = False
    args.offline_mode = False
    
    assert(os.path.exists(args.model_path)), f"Model file not found: {args.model_path}"
    assert(args.model_path.endswith(".pth")), f"Model file must be a .pth file: {args.model_path}"
    
    models_paths_list = [args.model_path]
    
    for model_path in models_paths_list:
        all_labels, all_probabilities, config = main(args, model_path)    


