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

from functools import partial

# Import custom utils
from utils.print_utils import *
from utils.loader_utils import load_hui_dataset
from utils.data_utils import VITPOSE_KEYPOINTS_NAMES
from utils.other_utils import read_yaml_to_dic
from utils.debug_utils import plot_input_tensor, update_old_config_dict, plot_input_tensors_skformer, export_unique_track_identifiers
from utils.eval_utils import get_best_threshold_f1
from utils.training_utils import get_lr_scheduler

# Ensure reproducibility with seeds and use the FIX_LIST as seed for the dataloaders to always sample the same tracks from one epoch to another
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
FIX_LIST = [3662, 5427, 3606, 3726, 3417, 6031, 7527, 1501, 4501, 9588, 2712, 4509, 2752, 57, 9256, 3417, 8694, 9336, 6870, 3587, 2675, 3613, 9281, 4883, 7570, 8967, 1654, 5194, 9746, 4310, 2848, 9954]

TRAIN_ITER_COUNT = 0
EVAL_ITER_COUNT = 0

def collate_fn_multidataset(batch):
    ''' 
    Collate function for the multidataset pretraining
    '''
    input_tensor = torch.stack([item[0] for item in batch])
    label = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    return input_tensor, label, metadata

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_epoch(model, dataloader, criterion, optimizer, device, wandb_logger=None, config=None):
    """Train the model for one epoch."""
    global TRAIN_ITER_COUNT
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    progress_bar = tqdm(dataloader, desc="Training")

    start.record()
    
    # for batch_idx, (input_tensor, label, metadata, images_tensor, masks_tensor) in enumerate(progress_bar):
    for batch_idx, (input_tensor, label, metadata) in enumerate(progress_bar):
        
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        
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
                        
            if False:
                plot_input_tensors_skformer(coco_tensor, sapiens_tensor, input_tensor, input_format_name)
        
        elif type(model) == STG_NF:
            
            input_tensor = input_tensor_to_format_by_channel(input_tensor, metadata, dataloader.dataset.data_columns_in_dataset) # B,T,V,C with V=17
            input_tensor = keypoints17_to_coco18_torch(input_tensor) # to Openpose format
            input_tensor = input_tensor.permute(0, 3, 1, 2).contiguous() # B,C,T,V with V=18
            # B,C,T,V = input_tensor.shape # with B=batch, C=channels=3, T=time, V=keypoints=18

            score = torch.ones((B,T)).to(device).amin(dim=-1)
            label_model = torch.where(label == 0, 1, -1).to(device) # Map label 0=normal=1, 1=abnormal=-1
            
            input_tensor = input_tensor[:, :2].float()
            model_kwargs["score"] = score
            model_kwargs["label"] = label_model
            
        # Move to device
        input_tensor = input_tensor.to(device)
        label = label.to(device).float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        model_output = model(input_tensor, **model_kwargs) # Note that in the case of STG_NF, the return is actually the nll itself
                
        loss = criterion(model_output.squeeze(), label) if type(model) != STG_NF else torch.mean(model_output)

        # Backward pass with optional gradient clipping
        loss.backward()
        if config["grad_clip"] > 0.0:
            # default clipping is 1.0 for Skateformer and 100 for STG_NF
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        
        if wandb_logger is not None:
            wandb_logger.log({
                'train/iter/batchloss': loss.item(),
                'train/iter/step': TRAIN_ITER_COUNT,
            })
            
        TRAIN_ITER_COUNT += B
        
        # Statistics
        total_loss += loss.item()
                
        # Collect predictions and probabilities for metrics
        if type(model) == STG_NF:
            with torch.no_grad():
                # Run a pass with only one labels (normal) to get the probabilities (actually the NLL wrt to ones labels)
                probabilities = model(input_tensor, label=torch.ones_like(label_model), score=score) # torch.tensor of shape (B,)
        else:
            probabilities = torch.sigmoid(model_output.squeeze()) # torch.tensor of shape (B,)

        # make a 1D tensor if probabilities is a 0D tensor (case of B=1)
        if probabilities.ndim == 0:
            probabilities = probabilities.unsqueeze(0) # torch.tensor of shape (1,)
        
        predictions = probabilities > 0.5
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_labels.extend(label.cpu().numpy().tolist())
        all_probabilities.extend(probabilities.detach().cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
            
    end.record()
    torch.cuda.synchronize()
    print(f"Time taken: {start.elapsed_time(end)} ms")
    
    # Calculate metrics
    if np.all([p == all_predictions[0] for p in all_predictions]):
        # prWarning(f"[Train] All predictions are the same @0.5 threshold : {all_predictions[0]}")
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    
    # Calculate F1 at 0.95 threshold
    predictions_at_095 = [1 if p > 0.95 else 0 for p in all_probabilities]
    if np.all([p == predictions_at_095[0] for p in predictions_at_095]):
        f1_at_095 = 0.0
    else:
        _, _, f1_at_095, _ = precision_recall_fscore_support(all_labels, predictions_at_095, average='binary')
        
    accuracy = accuracy_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probabilities)
    best_threshold, best_f1 = get_best_threshold_f1(all_labels, all_probabilities, thresholds_min_max=True)
    ap = average_precision_score(all_labels, all_probabilities)
        
    return total_loss / len(dataloader), accuracy, precision, recall, f1, auc, ap, best_threshold, best_f1, f1_at_095


def evaluate(model, dataloader, criterion, device, wandb_logger=None, config=None):
    """Evaluate the model on a dataset."""
    global EVAL_ITER_COUNT
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        # for batch_idx, (input_tensor, label, metadata, images_tensor, masks_tensor) in enumerate(progress_bar):
        for batch_idx, (input_tensor, label, metadata) in enumerate(progress_bar):

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
            
            loss = criterion(model_output.squeeze(), label) if type(model) != STG_NF else torch.mean(model_output)
             
            if wandb_logger is not None:
                wandb_logger.log({
                    'val/iter/batchloss': loss.item(),
                    'val/iter/step': EVAL_ITER_COUNT,
                })
            
            EVAL_ITER_COUNT += B
            
            # Statistics
            total_loss += loss.item()
            
            probabilities = torch.sigmoid(model_output.squeeze()) if type(model) != STG_NF else model_output # torch.tensor of shape (B,)

            # make a 1D tensor if probabilities is a 0D tensor (case of B=1)
            if probabilities.ndim == 0:
                probabilities = probabilities.unsqueeze(0) # torch.tensor of shape (1,)

            predictions = probabilities > 0.5
                
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(label.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    # Calculate metrics
    if np.all([p == all_predictions[0] for p in all_predictions]):
        # prWarning(f"[Train] All predictions are the same @0.5 threshold : {all_predictions[0]}")
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    
    # Calculate F1 at 0.95 threshold
    predictions_at_095 = [1 if p > 0.95 else 0 for p in all_probabilities]
    if np.all([p == predictions_at_095[0] for p in predictions_at_095]):
        f1_at_095 = 0.0
    else:
        _, _, f1_at_095, _ = precision_recall_fscore_support(all_labels, predictions_at_095, average='binary')
        
    accuracy = accuracy_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probabilities)
    best_threshold, best_f1 = get_best_threshold_f1(all_labels, all_probabilities, thresholds_min_max=True)
    ap = average_precision_score(all_labels, all_probabilities)
        
    return total_loss / len(dataloader), accuracy, precision, recall, f1, auc, ap, best_threshold, best_f1, f1_at_095


def train_model(args, num_workers=0):
    
    prInfo(f"Hostname: {hostname}")
    prInfo(f"use_wandb: {args.use_wandb}")
    prInfo(f"num_workers: {num_workers}")
    
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_name = f"{timestamp}_interaction"

    # Load config from YAML file if they exist
    assert(args.hp_config_file.endswith(".yaml")), f"Config dictionary must be a YAML file (got {args.hp_config_file})"
    assert(os.path.exists(args.hp_config_file)), f"Config dictionary file does not exist (at {args.hp_config_file})"
    config = read_yaml_to_dic(args.hp_config_file)
    train_model_type = config["force_model_type"]
    experiment_name = config["experiment_name"]
    prInfo(f"Using config dictionary from YAML file : {args.hp_config_file}")

    # for compatibility with old config dictionaries
    config = update_old_config_dict(config)

    # Check preload_data requirements
    if args.preload_data:
        if not config.get("fix_index_per_track_train", False) or not config.get("fix_index_per_track_val", False):
            prError("Preload data requires fix_index_per_track_train and fix_index_per_track_val to be True.")
            prError("This is because preloading saves a specific index for each track, and random sampling would invalidate the preloaded data.")
            exit(1)
        prInfo("Preload data mode enabled - will preprocess and cache datasets for faster training")

    experiments_dir = os.path.join(here, "experiments", "results", experiment_name+"_"+timestamp)
    if not os.path.exists(experiments_dir) and not args.preload_only:
        os.makedirs(experiments_dir)
    elif args.preload_only:
        prInfo(f"Preload only mode enabled - will not create experiments directory: {experiments_dir}")
    else:
        prError(f"Experiments directory already exists: {experiments_dir}")
        exit(1)
        
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    prInfo(f"Using device: {device}")
    
    # Initialize wandb
    if args.use_wandb and not args.preload_only:
        # Generate run name if not provided
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        wandb_run_name = f"{train_model_type.lower()}_interaction_{timestamp}"
        project_name = args.project_name + "hui-interaction-prediction" if args.project_name != "" else "hui-interaction-prediction"
        wandb_run = wandb.init(
            project=project_name,
            name=wandb_run_name,
            config={
                **config,
                "ncolumns": len(config["include_columns"]), # easier to see in wandb
                "device": str(device),
                "model_type": train_model_type.upper(),
                "task": "interaction_prediction"
            }
        )
        wandb_run.define_metric("train/epoch/*", step_metric="train/epoch/step")
        wandb_run.define_metric("train/iter/*", step_metric="train/iter/step")
        wandb_run.define_metric("val/epoch/*", step_metric="val/epoch/step")
        wandb_run.define_metric("val/iter/*", step_metric="val/iter/step")
        prSuccess(f"Wandb initialized: {wandb.run.url}")
    elif args.preload_only:
        prInfo("Preload only mode enabled - do not create wandb run")
        wandb_run = None
    else:
        wandb_run = None
        
    # Create datasets
    prInfo("Loading training dataset...")
    
    train_dataset = load_hui_dataset(args, config, split="train", num_workers=num_workers) # HUIInteract360 or HUIInteract360Light

    train_positives_tracks = train_dataset.total_positives_tracks
    train_negatives_tracks = train_dataset.total_negatives_tracks
    train_possible_positives_segments = train_dataset.total_possible_positives_segments
    train_possible_negatives_segments = train_dataset.total_possible_negatives_segments
    train_used_positive_segments = train_dataset.total_used_positive_segments
    train_used_negative_segments = train_dataset.total_used_negative_segments
    prInfo("Training dataset statistics:")
    print(f"\tTraining positives (number of tracks): {train_positives_tracks}")
    print(f"\tTraining negatives (number of tracks): {train_negatives_tracks}")
    print(f"\tTraining possible positives (number of segments): {train_possible_positives_segments}")
    print(f"\tTraining possible negatives (number of segments): {train_possible_negatives_segments}")
    print(f"\tTraining used positive segments: {train_used_positive_segments}")
    print(f"\tTraining used negative segments: {train_used_negative_segments}")
    
    
    print("\n\n")
    prInfo("Loading validation dataset...")
    
    val_dataset = load_hui_dataset(args, config, split="val", num_workers=num_workers) # HUIInteract360 or HUIInteract360Light
    
    val_positives_tracks = val_dataset.total_positives_tracks
    val_negatives_tracks = val_dataset.total_negatives_tracks
    val_possible_positives_segments = val_dataset.total_possible_positives_segments
    val_possible_negatives_segments = val_dataset.total_possible_negatives_segments
    val_used_positive_segments = val_dataset.total_used_positive_segments
    val_used_negative_segments = val_dataset.total_used_negative_segments
    prInfo("Validation dataset statistics:")
    print(f"\tValidation positives (number of tracks): {val_positives_tracks}")
    print(f"\tValidation negatives (number of tracks): {val_negatives_tracks}")
    print(f"\tValidation possible positives (number of segments): {val_possible_positives_segments}")
    print(f"\tValidation possible negatives (number of segments): {val_possible_negatives_segments}")
    print(f"\tValidation used positive segments: {val_used_positive_segments}")
    print(f"\tValidation used negative segments: {val_used_negative_segments}")
    
    # Export track identifiers if requested
    if args.export_tracks_ids:
        cross_eval_type = config.get("cross_eval_type", "unknown")
        prInfo("Exporting unique track identifiers...")
        export_unique_track_identifiers(train_dataset, split="train", cross_eval_type=cross_eval_type, add_label=True)
        export_unique_track_identifiers(val_dataset, split="val", cross_eval_type=cross_eval_type, add_label=True)
    
    if wandb_run is not None:
        wandb_run.log({
            "train_positives_tracks": train_positives_tracks,
            "train_negatives_tracks": train_negatives_tracks,
            "val_positives_tracks": val_positives_tracks,
            "val_negatives_tracks": val_negatives_tracks,
            
            "train_possible_positives_segments": train_possible_positives_segments,
            "train_possible_negatives_segments": train_possible_negatives_segments,
            "val_possible_positives_segments": val_possible_positives_segments,
            "val_possible_negatives_segments": val_possible_negatives_segments,
            
            "train_used_positive_segments": train_used_positive_segments,
            "train_used_negative_segments": train_used_negative_segments,
            "val_used_positive_segments": val_used_positive_segments,
            "val_used_negative_segments": val_used_negative_segments,
            
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        })
    
    
    # Get input dimensions
    input_dim = len(train_dataset.data_columns_in_dataset)
    print(f"Input dimensions: {input_dim}")
    print(train_dataset.data_columns_in_dataset)
    if config["use_polar_coordinates"] == True:
        prInfo("Using polar coordinates adding 2x the number of input dimensions for the x,y coordinates")
        for col in train_dataset.data_columns_in_dataset:
            if col.endswith("_x") or col.endswith("_y"):
                # use sin/cos embedding for each of the x and y coordinates so it doubles their number (but score, and non x,y coordinates remains the same)
                input_dim += 1
            if col == "xmin" or col == "xmax" or col == "ymin" or col == "ymax":
                # use sin/cos embedding for each of the x and y coordinates so it doubles their number (but score, and non x,y coordinates remains the same)
                input_dim += 1
    

    sequence_length = config["input_length_in_frames"] // config["subsample_frames"]
    
    prInfo(f"Input dimension: {input_dim}")
    prInfo(f"Sequence length: {sequence_length}")
    prInfo(f"Training samples: {len(train_dataset)}")
    prInfo(f"Validation samples: {len(val_dataset)}")
    
    if args.preload_only:
        prInfo("Preload only mode enabled - do not create models")
        return

    # Create data loaders
    g = torch.Generator()
    g.manual_seed(SEED)
    if num_workers == 0:
        num_workers = min(int(mp.cpu_count()-1), 32)
        prInfo(f"Using {num_workers} workers for training (CPU count: {mp.cpu_count()})")
        
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config["batch_size"], 
                                  shuffle=True, 
                                  num_workers=num_workers, 
                                  worker_init_fn=seed_worker, 
                                  generator=g, 
                                  drop_last = False,
                                  collate_fn=collate_fn_multidataset)
    
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=config["batch_size"], 
                                shuffle=False, 
                                num_workers=num_workers, 
                                worker_init_fn=seed_worker, 
                                generator=g, 
                                drop_last = False,
                                collate_fn=collate_fn_multidataset)
    
    if train_model_type.lower() == "mlp":
        model = MLPInteractionPredictor(
            input_dim=input_dim,
            sequence_length=sequence_length,
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"]
        ).to(device)
        
    elif train_model_type.lower() == "lstm":
        model = LSTMInteractionPredictor(
            input_dim=input_dim,
            sequence_length=sequence_length,
            hidden_dim=config["lstm_hidden_dim"],
            num_layers=config["lstm_num_layers"],
            dropout=config["lstm_dropout"],
            bidirectional=False
        ).to(device)
    
    elif train_model_type.lower() == "stg_nf":
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

    elif train_model_type.lower() == "stgcn":
        model = STGCN(
            in_channels=config["stgcn_in_channels"],
            num_class=1,
            graph_args={"layout": config["stgcn_layout"], "strategy": 'spatial'},
            edge_importance_weighting=config["stgcn_edge_importance_weighting"],
        ).to(device)
        
    elif train_model_type.lower() == "skateformer":
        
        # default params for nwucla
        # model_args:
        # num_classes: 10 # changed to 1
        # num_people: 1
        # num_points: 20
        # kernel_size: 7
        # num_heads: 32
        # attn_drop: 0.5
        # head_drop: 0.0
        # rel: True
        # drop_path: 0.2
        # type_1_size: [8, 4]
        # type_2_size: [8, 5]
        # type_3_size: [8, 4]
        # type_4_size: [8, 5]
        # mlp_ratio: 1.0
        # index_t: True
        
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
        

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prInfo(f"Model created with {millify(trainable_parameters)} trainable parameters")
    
    if wandb_run is not None:
        wandb_run.log({
            "trainable_parameters": trainable_parameters,
        })
    
    # Loss function and optimizer
    if config["loss_type"] == "BCEWithLogitsLoss":
        pos_weight = None
        if config["use_weighted_loss"]:
            if train_used_positive_segments is not None and train_used_negative_segments is not None:
                # Use the ratio of used positive and negative segments to compute the weight when using segments (with stride)
                pos_weight = torch.tensor(train_used_negative_segments / train_used_positive_segments) if train_used_positive_segments > 0 else torch.tensor(1.0)
            else:
                # Use the ratio of total positives and negatives to compute the weight when not using segments (without stride)
                pos_weight = torch.tensor(train_negatives_tracks / train_positives_tracks) if train_positives_tracks > 0 else torch.tensor(1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Loss type {config['loss_type']} not supported")
    
    if "weight_decay" in config and config["weight_decay"] is not None:
        prInfo(f"Using weight decay: {config['weight_decay']}")
        # ugly but a way to keep default value when none are specified in the yaml file
        if config["optimizer_type"] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        elif config["optimizer_type"] == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        elif config["optimizer_type"] == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        elif config["optimizer_type"] == "Adamax":
            optimizer = optim.Adamax(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        else:
            raise ValueError(f"Optimizer type {config['optimizer_type']} not supported")        
    else:
        prInfo("Using default weight decay in optimizers")
        if config["optimizer_type"] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        elif config["optimizer_type"] == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
        elif config["optimizer_type"] == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
        elif config["optimizer_type"] == "Adamax":
            optimizer = optim.Adamax(model.parameters(), lr=config["learning_rate"])
        else:
            raise ValueError(f"Optimizer type {config['optimizer_type']} not supported")
    
    lr_scheduler = get_lr_scheduler(optimizer, config)
    
    # Training loop
    best_train_auc = 0.0
    best_train_f1 = 0.0
    best_train_precision = 0.0
    best_train_recall = 0.0
    best_val_auc = 0.0
    best_val_f1 = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0
    best_train_f1_with_adaptative_threshold = 0.0
    best_val_f1_with_adaptative_threshold = 0.0
    best_train_ap = 0.0
    best_val_ap = 0.0
    best_train_f1_at_095 = 0.0
    best_val_f1_at_095 = 0.0
    train_losses = []
    val_losses = []
    val_aucs = []
    
    prInfo("Starting training...")
    start_time = time.time()
    
    total_epochs = config["epochs"]

    new_lr = config["learning_rate"] # no update, default value 
    for epoch in range(total_epochs):

        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc, train_ap, train_f1_threshold, train_f1_with_adaptative_threshold, train_f1_at_095 = train_epoch(
            model, train_dataloader, criterion, optimizer, device, wandb_logger=wandb_run, config=config
        )

        # Evaluate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_ap, val_f1_threshold, val_f1_with_adaptative_threshold, val_f1_at_095 = evaluate(
            model, val_dataloader, criterion, device, wandb_logger=wandb_run, config=config
        )

        # Update LR
        if lr_scheduler is not None:
            if config["lr_scheduler_type"] == "CosineAnnealingWithWarmup":
                lr_scheduler.step(epoch)
                new_lr = lr_scheduler._get_values(epoch)[0]
            else:
                lr_scheduler.step()
                new_lr = lr_scheduler.get_last_lr()[0]
            prInfo(f"New learning rate: {new_lr:.7f}")
            
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        
        # Update best metrics
        if train_auc > best_train_auc:
            best_train_auc = train_auc
        if train_f1 > best_train_f1:
            best_train_f1 = train_f1
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        if train_prec > best_train_precision:
            best_train_precision = train_prec
        if train_rec > best_train_recall:
            best_train_recall = train_rec
        if val_prec > best_val_precision:
            best_val_precision = val_prec
        if val_rec > best_val_recall:
            best_val_recall = val_rec
            
        if train_f1_with_adaptative_threshold > best_train_f1_with_adaptative_threshold:
            best_train_f1_with_adaptative_threshold = train_f1_with_adaptative_threshold
        if val_f1_with_adaptative_threshold > best_val_f1_with_adaptative_threshold:
            best_val_f1_with_adaptative_threshold = val_f1_with_adaptative_threshold
        
        if train_ap > best_train_ap:
            best_train_ap = train_ap
        if val_ap > best_val_ap:
            best_val_ap = val_ap
        
        if train_f1_at_095 > best_train_f1_at_095:
            best_train_f1_at_095 = train_f1_at_095
        if val_f1_at_095 > best_val_f1_at_095:
            best_val_f1_at_095 = val_f1_at_095
        
        # Print metrics
        prInfo(f"[Epoch {epoch+1}/{total_epochs}] Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, AP: {train_ap:.4f}, Best F1 @{train_f1_threshold:.4f} threshold: {train_f1_with_adaptative_threshold:.4f}, F1@0.95: {train_f1_at_095:.4f}")
        prInfo(f"[Epoch {epoch+1}/{total_epochs}] Validation  - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}, AP: {val_ap:.4f}, Best F1 @{val_f1_threshold:.4f} threshold: {val_f1_with_adaptative_threshold:.4f}, F1@0.95: {val_f1_at_095:.4f}")

        save_model_name = f"{train_model_type.lower()}_interaction_model.pth"
        
        # Save best model (based on validation adaptative F1)
        if (val_f1_with_adaptative_threshold >= best_val_f1_with_adaptative_threshold) or (val_ap >= best_val_ap) or (val_auc >= best_val_auc) or (epoch == total_epochs - 1):
            suffixes = []
            if val_f1_with_adaptative_threshold >= best_val_f1_with_adaptative_threshold:
                suffixes.append(f"_best_adaptative_f1.pth")
            if val_ap >= best_val_ap:
                suffixes.append(f"_best_ap.pth")
            if val_auc >= best_val_auc:
                suffixes.append(f"_best_auc.pth")
            if epoch == total_epochs - 1:
                suffixes.append(f"_last.pth")
            
            for suffix in suffixes:
                if args.save_model:
                    torch.save({
                        'epoch': epoch,
                        "current_lr": new_lr,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_auc': val_auc,
                        'val_loss': val_loss,
                        'val_ap': val_ap,
                        'val_f1': val_f1,
                        'val_f1_at_095': val_f1_at_095,
                        'val_f1_with_adaptative_threshold': val_f1_with_adaptative_threshold,
                        'val_f1_used_adaptative_threshold': val_f1_threshold,
                        'best_train_auc': best_train_auc,
                        'best_train_f1': best_train_f1,
                        'best_train_ap': best_train_ap,
                        'best_val_auc': best_val_auc,
                        'best_val_f1': best_val_f1,
                        'best_train_precision': best_train_precision,
                        'best_train_recall': best_train_recall,
                        'best_val_precision': best_val_precision,
                        'best_val_recall': best_val_recall,
                        'best_train_f1_with_adaptative_threshold': best_train_f1_with_adaptative_threshold,
                        'best_val_f1_with_adaptative_threshold': best_val_f1_with_adaptative_threshold,
                        'best_val_f1_at_095': best_val_f1_at_095,
                        'best_train_f1_at_095': best_train_f1_at_095,
                        'args': args,
                        'config': config
                    }, os.path.join(experiments_dir, save_model_name.replace(".pth", suffix)))
                    prSuccess(f"New best model saved with {save_model_name.replace('.pth', suffix)}")

        # Log metrics to wandb
        if args.use_wandb:
            wandb_run.log({
                # "epoch": epoch + 1,
                "train/epoch/step": epoch,
                "val/epoch/step": epoch,
                "train/epoch/loss": train_loss,
                "train/epoch/accuracy": train_acc,
                "train/epoch/precision": train_prec,
                "train/epoch/recall": train_rec,
                "train/epoch/f1": train_f1,
                "train/epoch/auc": train_auc,
                "train/epoch/f1_with_adaptative_threshold": train_f1_with_adaptative_threshold,
                "train/epoch/f1_used_adaptative_threshold": train_f1_threshold,
                "train/epoch/lr": new_lr,
                "train/epoch/ap": train_ap,
                "train/epoch/f1_at_095": train_f1_at_095,
                "val/epoch/loss": val_loss,
                "val/epoch/accuracy": val_acc,
                "val/epoch/precision": val_prec,
                "val/epoch/recall": val_rec,
                "val/epoch/f1": val_f1,
                "val/epoch/auc": val_auc,
                "val/epoch/f1_with_adaptative_threshold": val_f1_with_adaptative_threshold,
                "val/epoch/f1_used_adaptative_threshold": val_f1_threshold,
                "val/epoch/ap": val_ap,
                "val/epoch/f1_at_095": val_f1_at_095,
                "train/epoch/best_train_auc": best_train_auc,
                "train/epoch/best_train_f1": best_train_f1,
                "train/epoch/best_train_ap": best_train_ap,
                "train/epoch/best_train_f1_at_095": best_train_f1_at_095,
                "val/epoch/best_val_auc": best_val_auc,
                "val/epoch/best_val_f1": best_val_f1,
                "val/epoch/best_val_ap": best_val_ap,
                "val/epoch/best_val_f1_at_095": best_val_f1_at_095,
                "train/epoch/best_train_precision": best_train_precision,
                "train/epoch/best_train_recall": best_train_recall,
                "val/epoch/best_val_precision": best_val_precision,
                "val/epoch/best_val_recall": best_val_recall,
                "train/epoch/best_train_f1_with_adaptative_threshold": best_train_f1_with_adaptative_threshold,
                "val/epoch/best_val_f1_with_adaptative_threshold": best_val_f1_with_adaptative_threshold,
            })
            
    total_time = time.time() - start_time
    prSuccess(f"\nTraining completed in {total_time:.2f} seconds")
    
    print("--------------------------------")
    prSuccess(f"Best train AUC: {best_train_auc:.4f}")
    prSuccess(f"Best train AP: {best_train_ap:.4f}")
    prSuccess(f"Best validation AUC: {best_val_auc:.4f}")
    prSuccess(f"Best validation AP: {best_val_ap:.4f}")
    
    print("--------------------------------")
    prSuccess(f"Last train AUC: {train_auc:.4f}")
    prSuccess(f"Last train AP: {train_ap:.4f}")
    prSuccess(f"Last validation AUC: {val_auc:.4f}")
    prSuccess(f"Last validation AP: {val_ap:.4f}")

    if args.use_wandb:
        # Finish wandb run
        wandb_run.finish()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train MLP for interaction prediction')
    parser.add_argument("--device", default="auto", type=str, help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--save_model", "-sm", action="store_true", default=False, help="Save the trained model")
    parser.add_argument("--use_wandb", "-uw", action="store_true", default=False, help="Use Weights & Biases logging")
    parser.add_argument("--hp_config_file", "-hp", required=False, type=str, default="./experiments/configs/expe_classifier/mlp_config.yaml", help="Config configuration file path")
    parser.add_argument("--expe_iterations_from", "-eif", default=0, type=int, help="Start from the nth iteration of the experiment")
    parser.add_argument("--expe_iterations_to", "-eit", default=None, type=int, help="End at the nth iteration of the experiment")
    parser.add_argument("--project_name", "-pn", default="", type=str, help="Project name")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Verbose mode")
    parser.add_argument("--num_workers", "-nw", default=0, type=int, help="Number of workers for dataloader (default: auto)")
    parser.add_argument("--hf_local_dir", "-hld", default="default", type=str, help="HF local directory (default: default, ie ./datasets/hf_data)")
    parser.add_argument("--preload_data", "-pd", action="store_true", default=False, help="Preload data to speed up training (requires fix_index_per_track to be True)")
    parser.add_argument("--preload_only", "-po", action="store_true", default=False, help="Preload data only, do not train")
    parser.add_argument("--offline_mode", "-om", action="store_true", default=False, help="Offline mode, do not download from Hugging Face")
    parser.add_argument("--export_tracks_ids", "-eti", action="store_true", default=False, help="Export unique track identifiers for train and validation sets to txt files")
    args = parser.parse_args()
    
    if args.use_wandb:
        import wandb
        wandb.login()
    
    configs_to_run = []

    if os.path.isdir(args.hp_config_file):
        # is a dir check if valid then run for each
        yaml_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(args.hp_config_file)) for f in fn if f.endswith(".yaml")]
        yaml_files.sort()
        prInfo(f"Found {len(yaml_files)} YAML files in {args.hp_config_file}, running multiple experiments")
        for yaml_file in yaml_files:
            configs_to_run.append(yaml_file)
    elif os.path.isfile(args.hp_config_file) and args.hp_config_file.endswith(".yaml"):
        configs_to_run = [args.hp_config_file]
    elif args.hp_config_file == "":
        configs_to_run = [None]
    else:
        raise ValueError(f"Invalid config configuration file: {args.hp_config_file} (either an experiment dir or a YAML file)")
    
    if len(configs_to_run) == 1:
        # Single config, run normally
        prInfo(f"Running single experiment with {configs_to_run[0]}")
        args.hp_config_file = configs_to_run[0]
        
        tic = time.time()
        train_model(args, num_workers=args.num_workers)
        toc = time.time()
        prTimer("Trained model in", tic, toc)

    elif len(configs_to_run) > 1:
        prInfo(f"Running {len(configs_to_run)} experiments sequentially")
        
        if args.expe_iterations_from > 0 and args.expe_iterations_to is None:
            prInfo(f"Skipping the first {args.expe_iterations_from} iterations of the experiment")
            configs_to_run = configs_to_run[args.expe_iterations_from:]
        elif args.expe_iterations_from > 0 and args.expe_iterations_to is not None:
            prInfo(f"Skipping the first {args.expe_iterations_from} iterations of the experiment and running until the {args.expe_iterations_to}th iteration")
            configs_to_run = configs_to_run[args.expe_iterations_from:args.expe_iterations_to]
        elif args.expe_iterations_from == 0 and args.expe_iterations_to is not None:
            prInfo(f"Running until the {args.expe_iterations_to}th iteration")
            configs_to_run = configs_to_run[:args.expe_iterations_to]
            
        print("--------------------------------")
        for config in configs_to_run:
            print(config)
        print("--------------------------------")
        
        failed_configs = []
        for config in configs_to_run:
            prInfo(f"Running experiment with {config}")
            args.hp_config_file = config
            
            # try:
            train_model(args, num_workers=args.num_workers)
            # except Exception as e:
            #     prError(f"Error training model, skipping this experiment: {e}")
            #     failed_configs.append(config)
            #     continue
                
        if len(failed_configs) > 0:
            prWarning(f"Failed to train {len(failed_configs)} experiments: {failed_configs}")
        else:
            prSuccess("All experiments completed successfully!")
    
