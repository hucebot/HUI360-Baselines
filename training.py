# Import standard libraries
import argparse
import os
# Define the here variable to be the directory of the current file
here = os.path.dirname(os.path.abspath(__file__))
from tqdm import tqdm
import joblib
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
from sklearn.ensemble import RandomForestClassifier

# Import the datasets classes
# when using strictly classical training
from datasets.HUIDataset import HUIInteract360

# Import the predictors classes
from predictors.mlp import MLPInteractionPredictor
from predictors.lstm import LSTMInteractionPredictor

# Import custom utils
from utils.print_utils import *
from utils.other_utils import read_yaml_to_dic

# Ensure reproducibility with seeds and use the FIX_LIST as seed for the dataloaders to always sample the same tracks from one epoch to another
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
FIX_LIST = [3662, 5427, 3606, 3726, 3417, 6031, 7527, 1501, 4501, 9588, 2712, 4509, 2752, 57, 9256, 3417, 8694, 9336, 6870, 3587, 2675, 3613, 9281, 4883, 7570, 8967, 1654, 5194, 9746, 4310, 2848, 9954]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    

def get_best_threshold_f1(all_labels, all_probabilities):
    """Get the best threshold for the model."""
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.0
    best_f1 = 0.0
    for threshold in thresholds:
        predictions = (all_probabilities >= threshold).astype(int)
        if np.all([p == predictions[0] for p in predictions]):
            continue
        _, _, f1, _ = precision_recall_fscore_support(all_labels, predictions, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (input_tensor, label, metadata) in enumerate(progress_bar):
        
        # Move to device
        input_tensor = input_tensor.to(device)
        label = label.to(device).float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        logits = model(input_tensor)
        loss = criterion(logits.squeeze(), label)

        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        
        # Collect predictions and probabilities for metrics
        probabilities = torch.sigmoid(logits.squeeze())
        predictions = probabilities > 0.5
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_labels.extend(label.cpu().numpy().tolist())
        all_probabilities.extend(probabilities.detach().cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    # Calculate metrics
    if np.all([p == all_predictions[0] for p in all_predictions]):
        prWarning(f"[Train] All predictions are the same : {all_predictions[0]}")
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        auc = 0.0
        ap = 0.0
    else:
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        auc = roc_auc_score(all_labels, all_probabilities)
        best_threshold, best_f1 = get_best_threshold_f1(all_labels, all_probabilities)
        ap = average_precision_score(all_labels, all_probabilities)
        
    return total_loss / len(dataloader), accuracy, precision, recall, f1, auc, ap, best_threshold, best_f1


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch_idx, (input_tensor, label, metadata) in enumerate(progress_bar):
            # Move to device
            input_tensor = input_tensor.to(device)
            label = label.to(device).float()
            
            # Forward pass
            logits = model(input_tensor)
            loss = criterion(logits.squeeze(), label)
            
            # Statistics
            total_loss += loss.item()
            
            # Collect predictions and probabilities
            probabilities = torch.sigmoid(logits.squeeze())
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
        # all the same label
        prWarning(f"[Evaluate] All predictions are the same : {all_predictions[0]}")
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        auc = 0.0
        ap = 0.0
    else:
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        auc = roc_auc_score(all_labels, all_probabilities)
        best_threshold, best_f1 = get_best_threshold_f1(all_labels, all_probabilities)
        ap = average_precision_score(all_labels, all_probabilities)
        
    return total_loss / len(dataloader), accuracy, precision, recall, f1, auc, ap, best_threshold, best_f1



def train_model(args, num_workers=None, model_type = "mlp"):
    
    prInfo(f"Hostname: {hostname}")
    prInfo(f"use_wandb: {args.use_wandb}")
    prInfo(f"num_workers: {num_workers}")
    prInfo(f"model_type: {model_type}")
    
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_name = f"{timestamp}_{model_type.lower()}_interaction"

    # Load hyperparameters from YAML file if they exist
    assert(args.hp_config_file.endswith(".yaml")), f"Hyperparameters dictionary must be a YAML file (got {args.hp_config_file})"
    assert(os.path.exists(args.hp_config_file)), f"Hyperparameters dictionary file does not exist (at {args.hp_config_file})"
    config = read_yaml_to_dic(args.hp_config_file)
    args.model_type = config["force_model_type"]
    experiment_name = config["experiment_name"]
    prInfo(f"Using hyperparameters dictionary from YAML file : {args.hp_config_file}")

    # Make some checks on hyperparameters and print warnings or raise errors
    if args.model_type.lower() == "rf" and (config["fix_index_per_track_train"] == False or config["fix_index_per_track_val"] == False):
        prError("RF training requires fix_index_per_track_train and fix_index_per_track_val to be True (no multiple epochs training). Enforcing it.")
        config["fix_index_per_track_train"] = True
        config["fix_index_per_track_list_train"] = FIX_LIST
        config["fix_index_per_track_val"] = True
        config["fix_index_per_track_list_val"] = FIX_LIST

    experiments_dir = os.path.join(here, "experiments", experiment_name+"_"+timestamp)
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    else:
        prError(f"Experiments directory already exists: {experiments_dir}")
        exit(1)
        
    # Set device
    if args.model_type.lower() == "rf":
        prInfo("Using CPU for RF training")
        device = torch.device("cpu")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    prInfo(f"Using device: {device}")
    
    # Initialize wandb
    if args.use_wandb:
        # Generate run name if not provided
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        wandb_run_name = f"{model_type.lower()}_interaction_{timestamp}"
        project_name = "hui-interaction-prediction-extmet"
        wandb_run = wandb.init(
            project=project_name,
            name=wandb_run_name,
            config={
                **config,
                "ncolumns": len(config["include_columns"]), # easier to see in wandb
                "device": str(device),
                "model_type": model_type.upper(),
                "task": "interaction_prediction"
            }
        )
        prSuccess(f"Wandb initialized: {wandb.run.url}")
    
    # Create datasets
    train_dataset = HUIInteract360(
        include_recordings=config["include_recordings_train"], 
        include_tracks="all", 
        include_columns=config["include_columns"],
        positive_cutoff=config["positive_cutoff_train"],
        interaction_cutoff=config["interaction_cutoff_train"],
        fixed_input_length=config["fixed_input_length"],
        input_length_in_frames=config["input_length_in_frames"],
        min_length_in_frames=config["min_length_in_frames"],
        max_length_in_frames=config["max_length_in_frames"],
        subsample_frames=config["subsample_frames"],
        min_keypoints_filter=config["min_keypoints_filter"],
        additional_filtering_dict=config["additional_filtering_dict"],
        return_images=False,
        return_masks=False,
        normalize_in_image=config["normalize_in_image"],
        normalize_keypoints_in_box=config["normalize_keypoints_in_box"],
        standardize_data=config["standardize_data"],
        fix_index_per_track=config["fix_index_per_track_train"],
        fix_index_per_track_list=config["fix_index_per_track_list_train"],
        force_positive_samples=config["force_positive_samples"],
        ignore_negative_tracks_after_biggest_mask_size=config["ignore_negative_tracks_after_biggest_mask_size_train"],
        force_aligment_with_biggest_mask_size=config["force_aligment_with_biggest_mask_size_train"],
        verbose=False,
        dataset_revision=config["dataset_revision"],
    )
    
    train_positives = train_dataset.total_positives
    train_negatives = train_dataset.total_negatives
    prInfo(f"Training positives: {train_positives}")
    prInfo(f"Training negatives: {train_negatives}")
    
    val_dataset = HUIInteract360(
        include_recordings=config["include_recordings_val"], 
        include_tracks="all", 
        include_columns=config["include_columns"],
        positive_cutoff=config["positive_cutoff_val"],    
        interaction_cutoff=config["interaction_cutoff_val"],
        fixed_input_length=config["fixed_input_length"],
        input_length_in_frames=config["input_length_in_frames"],
        min_length_in_frames=config["min_length_in_frames"],
        max_length_in_frames=config["max_length_in_frames"],
        subsample_frames=config["subsample_frames"],
        min_keypoints_filter=config["min_keypoints_filter"],
        additional_filtering_dict=config["additional_filtering_dict"],
        return_images=False,
        return_masks=False,
        normalize_in_image=config["normalize_in_image"],
        normalize_keypoints_in_box=config["normalize_keypoints_in_box"],
        standardize_data=config["standardize_data"],
        fix_index_per_track=config["fix_index_per_track_val"],
        fix_index_per_track_list=config["fix_index_per_track_list_val"],
        force_positive_samples=config["force_positive_samples"],
        ignore_negative_tracks_after_biggest_mask_size=config["ignore_negative_tracks_after_biggest_mask_size_val"],
        force_aligment_with_biggest_mask_size=config["force_aligment_with_biggest_mask_size_val"],
        verbose=False,
        dataset_revision=config["dataset_revision"],
    )
    
    val_positives = val_dataset.total_positives
    val_negatives = val_dataset.total_negatives
    prInfo(f"Validation positives: {val_positives}")
    prInfo(f"Validation negatives: {val_negatives}")
    
    # Get input dimensions
    input_dim = len(train_dataset.data_columns_in_dataset)
    sequence_length = config["input_length_in_frames"] // config["subsample_frames"]
    
    prInfo(f"Input dimension: {input_dim}")
    prInfo(f"Sequence length: {sequence_length}")
    prInfo(f"Training samples: {len(train_dataset)}")
    prInfo(f"Validation samples: {len(val_dataset)}")
    
    if args.model_type.lower() != "rf":
        # Create data loaders
        g = torch.Generator()
        g.manual_seed(SEED)
        if num_workers is None:
            num_workers = min(int(mp.cpu_count()-1), 32)
            prInfo(f"Using {num_workers} workers for training (CPU count: {mp.cpu_count()})")
        train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g, drop_last = True)
        val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g, drop_last = False)
        
        if args.model_type.lower() == "mlp":
            # Create model
            model = MLPInteractionPredictor(
                input_dim=input_dim,
                sequence_length=sequence_length,
                hidden_dims=config["hidden_dims"],
                dropout=config["dropout"]
            ).to(device)
        elif args.model_type.lower() == "lstm":
            # Create model
            model = LSTMInteractionPredictor(
                input_dim=input_dim,
                sequence_length=sequence_length,
                hidden_dim=config["lstm_hidden_dim"],
                num_layers=config["lstm_num_layers"],
                dropout=config["lstm_dropout"],
                bidirectional=False
            ).to(device)
            
        prInfo(f"Model created with {millify(sum(p.numel() for p in model.parameters() if p.requires_grad))} trainable parameters")
        
        # Loss function and optimizer
        if config["loss_type"] == "BCEWithLogitsLoss":
            pos_weight = None
            if config["use_weighted_loss"]:
                pos_weight = torch.tensor(train_negatives / train_positives)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            raise ValueError(f"Loss type {config['loss_type']} not supported")
        
        if config["optimizer_type"] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        elif config["optimizer_type"] == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
        elif config["optimizer_type"] == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
        else:
            raise ValueError(f"Optimizer type {config['optimizer_type']} not supported")
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        criterion = None
        optimizer = None
        
        # Load data for RF training
        train_X = []
        train_X_before_flattening = []
        train_y = []
        metadata_list = []
        for input_tensor, label, metadata in train_dataset:
            train_X_before_flattening.append(input_tensor)
            train_X.append(input_tensor.flatten())
            metadata_list.append(metadata)
            train_y.append(label)
        train_X = torch.stack(train_X).cpu().numpy()
        prInfo(f"Train X shape: {train_X.shape}")
        train_y = torch.stack(train_y).cpu().numpy()
        prInfo(f"Train y shape: {train_y.shape}")
        
        # Load data for RF validation
        val_X = []
        val_X_before_flattening = []
        val_y = []
        metadata_list = []
        for input_tensor, label, metadata in val_dataset:
            val_X_before_flattening.append(input_tensor)
            val_X.append(input_tensor.flatten())
            metadata_list.append(metadata)
            val_y.append(label)
        val_X = torch.stack(val_X).cpu().numpy()
        prInfo(f"Val X shape: {val_X.shape}")
        val_y = torch.stack(val_y).cpu().numpy()
        prInfo(f"Val y shape: {val_y.shape}")
        
        # Create RF model
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            class_weight=config["class_weight"],
            n_jobs = -1,
            random_state = SEED
        )
        
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
    train_losses = []
    val_losses = []
    val_aucs = []
    
    prInfo("Starting training...")
    start_time = time.time()
    
    total_epochs = config["epochs"] if args.model_type.lower() != "rf" else 1
    
    for epoch in range(total_epochs):
        # prInfo(f"\nEpoch {epoch+1}/{total_epochs}")
        
        if args.model_type.lower() != "rf":
            # Train
            train_loss, train_acc, train_prec, train_rec, train_f1, train_auc, train_ap, train_f1_threshold, train_f1_with_adaptative_threshold = train_epoch(
                model, train_dataloader, criterion, optimizer, device
            )

            # Evaluate
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_ap, val_f1_threshold, val_f1_with_adaptative_threshold = evaluate(
                model, val_dataloader, criterion, device
            )
            
        elif args.model_type.lower() == "rf":
            # Train
            model.fit(train_X, train_y)
            
            # Predict for training set
            train_predictions_proba = model.predict_proba(train_X)
            # keep only the positive class probability
            train_predictions_proba = train_predictions_proba[:, 1]
            train_predictions = (train_predictions_proba > 0.5).astype(np.int32)
            train_loss = 0.0 # no loss for RF
            if np.all([p == train_predictions[0] for p in train_predictions]):
                prWarning(f"[Train] All predictions are the same : {train_predictions[0]}")
                train_acc = 0.0
                train_prec = 0.0
                train_rec = 0.0
                train_f1 = 0.0
                train_auc = 0.0
                train_ap = 0.0
            else:
                train_acc = accuracy_score(train_y, train_predictions)
                train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(train_y, train_predictions, average='binary')
                train_auc = roc_auc_score(train_y, train_predictions_proba)
                train_ap = average_precision_score(train_y, train_predictions_proba)
                train_f1_threshold, train_f1_with_adaptative_threshold = get_best_threshold_f1(train_y, train_predictions_proba)
            
            # Predict for validation set
            val_predictions_proba = model.predict_proba(val_X)
            # keep only the positive class probability
            val_predictions_proba = val_predictions_proba[:, 1]
            val_predictions = (val_predictions_proba > 0.5).astype(np.int32)
            val_loss = 0.0 # no loss for RF
            if np.all([p == val_predictions[0] for p in val_predictions]):
                prWarning(f"[Validation] All predictions are the same : {val_predictions[0]}")
                val_acc = 0.0
                val_prec = 0.0
                val_rec = 0.0
                val_f1 = 0.0
                val_auc = 0.0
                val_ap = 0.0
            else:
                val_acc = accuracy_score(val_y, val_predictions)
                val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(val_y, val_predictions, average='binary')
                val_auc = roc_auc_score(val_y, val_predictions_proba)
                val_ap = average_precision_score(val_y, val_predictions_proba)
                val_f1_threshold, val_f1_with_adaptative_threshold = get_best_threshold_f1(val_y, val_predictions_proba)
        
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
        
        # Print metrics
        prInfo(f"[Epoch {epoch+1}/{total_epochs}] Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, AP: {train_ap:.4f}, Best F1 @{train_f1_threshold:.4f} threshold: {train_f1_with_adaptative_threshold:.4f}")
        prInfo(f"[Epoch {epoch+1}/{total_epochs}] Validation  - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}, AP: {val_ap:.4f}, Best F1 @{val_f1_threshold:.4f} threshold: {val_f1_with_adaptative_threshold:.4f}")

        save_model_name = f"{args.model_type.lower()}_interaction_model.pth"
        
        # Save best model (based on validation adaptative F1)
        if val_f1_with_adaptative_threshold >= best_val_f1_with_adaptative_threshold:
            if args.save_model and not args.model_type.lower() == "rf":
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_loss': val_loss,
                    'val_ap': val_ap,
                    'val_f1_with_adaptative_threshold': val_f1_with_adaptative_threshold,
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
                    'args': args,
                    'hyperparameters': config
                }, os.path.join(experiments_dir, save_model_name.replace(".pth", "_best_adaptative_f1.pth")))
                prSuccess(f"New best model saved with adaptative F1: {val_f1_with_adaptative_threshold:.4f}")
                
            elif args.save_model and args.model_type.lower() == "rf":
                save_hp_name = save_model_name
                save_forest_name = save_model_name.replace(".pth", "_forest.joblib")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': None,
                    'optimizer_state_dict': None,
                    'val_auc': val_auc,
                    'val_loss': val_loss,
                    'val_ap': val_ap,
                    'val_f1_with_adaptative_threshold': val_f1_with_adaptative_threshold,
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
                    'args': args,
                    'hyperparameters': config
                }, os.path.join(experiments_dir, save_hp_name.replace(".pth", "_best_adaptative_f1.pth")))
                joblib.dump(model, os.path.join(experiments_dir, save_forest_name.replace(".joblib", "_best_adaptative_f1.joblib")))
                prSuccess(f"New best model saved with adaptative F1: {val_f1_with_adaptative_threshold:.4f}")

        # Save best model (based on validation AP)
        if val_ap >= best_val_ap:
            if args.save_model and not args.model_type.lower() == "rf":
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_loss': val_loss,
                    'val_ap': val_ap,
                    'val_f1_with_adaptative_threshold': val_f1_with_adaptative_threshold,
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
                    'args': args,
                    'hyperparameters': config
                }, os.path.join(experiments_dir, save_model_name.replace(".pth", "_best_ap.pth")))
                prSuccess(f"New best model saved with AP: {val_ap:.4f}")
                
            elif args.save_model and args.model_type.lower() == "rf":
                save_hp_name = save_model_name
                save_forest_name = save_hp_name.replace(".pth", "_forest.joblib")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': None,
                    'optimizer_state_dict': None,
                    'val_auc': val_auc,
                    'val_loss': val_loss,
                    'val_ap': val_ap,
                    'val_f1_with_adaptative_threshold': val_f1_with_adaptative_threshold,
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
                    'args': args,
                    'hyperparameters': config
                }, os.path.join(experiments_dir, save_hp_name.replace(".pth", "_best_ap.pth")))
                joblib.dump(model, os.path.join(experiments_dir, save_forest_name.replace(".joblib", "_best_ap.joblib")))
                prSuccess(f"New best model saved with AP: {val_ap:.4f}")

        # Save best model (based on validation AUC)
        if val_auc >= best_val_auc:
            if args.save_model and not args.model_type.lower() == "rf":
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_loss': val_loss,
                    'val_ap': val_ap,
                    'val_f1_with_adaptative_threshold': val_f1_with_adaptative_threshold,
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
                    'args': args,
                    'hyperparameters': config
                }, os.path.join(experiments_dir, save_model_name.replace(".pth", "_best_auc.pth")))
                prSuccess(f"New best model saved with AUC: {val_auc:.4f}")
                
            elif args.save_model and args.model_type.lower() == "rf":
                save_hp_name = save_model_name
                save_forest_name = save_hp_name.replace(".pth", "_forest.joblib")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': None,
                    'optimizer_state_dict': None,
                    'val_auc': val_auc,
                    'val_loss': val_loss,
                    'val_ap': val_ap,
                    'val_f1_with_adaptative_threshold': val_f1_with_adaptative_threshold,
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
                    'args': args,
                    'hyperparameters': config
                }, os.path.join(experiments_dir, save_hp_name.replace(".pth", "_best_auc.pth")))
                joblib.dump(model, os.path.join(experiments_dir, save_forest_name.replace(".joblib", "_best_auc.joblib")))
                prSuccess(f"New best model saved with AUC: {val_auc:.4f}")

        # Log metrics to wandb
        if args.use_wandb:
            wandb_run.log({
                # "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "train/precision": train_prec,
                "train/recall": train_rec,
                "train/f1": train_f1,
                "train/auc": train_auc,
                "train/f1_with_adaptative_threshold": train_f1_with_adaptative_threshold,
                "train/f1_used_adaptative_threshold": train_f1_threshold,
                "train/ap": train_ap,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/precision": val_prec,
                "val/recall": val_rec,
                "val/f1": val_f1,
                "val/auc": val_auc,
                "val/f1_with_adaptative_threshold": val_f1_with_adaptative_threshold,
                "val/f1_used_adaptative_threshold": val_f1_threshold,
                "val/ap": val_ap,
                "best_train_auc": best_train_auc,
                "best_train_f1": best_train_f1,
                "best_train_ap": best_train_ap,
                "best_val_auc": best_val_auc,
                "best_val_f1": best_val_f1,
                "best_val_ap": best_val_ap,
                "best_train_precision": best_train_precision,
                "best_train_recall": best_train_recall,
                "best_val_precision": best_val_precision,
                "best_val_recall": best_val_recall,
                "best_train_f1_with_adaptative_threshold": best_train_f1_with_adaptative_threshold,
                "best_val_f1_with_adaptative_threshold": best_val_f1_with_adaptative_threshold,
            })
            
    total_time = time.time() - start_time
    prSuccess(f"\nTraining completed in {total_time:.2f} seconds")
    prSuccess(f"Best train AUC: {best_train_auc:.4f}")
    prSuccess(f"Best validation AUC: {best_val_auc:.4f}")
    # prSuccess(f"Best train F1: {best_train_f1:.4f}")
    # prSuccess(f"Best validation F1: {best_val_f1:.4f}")
    prSuccess(f"Best train AP: {best_train_ap:.4f}")
    prSuccess(f"Best validation AP: {best_val_ap:.4f}")
    # prSuccess(f"Best train F1 with adaptative threshold: {best_train_f1_with_adaptative_threshold:.4f}")
    # prSuccess(f"Best validation F1 with adaptative threshold: {best_val_f1_with_adaptative_threshold:.4f}")
    
    if args.use_wandb:
        # Finish wandb run
        wandb_run.finish()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train MLP for interaction prediction')
    parser.add_argument("--device", default="auto", type=str, help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--save_model", "-sm", action="store_true", default=False, help="Save the trained model")
    parser.add_argument("--use_wandb", default=False, type=bool, help="Use Weights & Biases logging")
    parser.add_argument("--model_type", "-mt", default="MLP", type=str, help="Model type to use (MLP, LSTM, RF). Will be forced changed by yaml config if necessary.")
    parser.add_argument("--hp_config_file", "-hp", required=False, type=str, default="./experiments/configs/expe_classifier/mlp_config.yaml", help="Hyperparameters configuration file path")
    parser.add_argument("--light_dataset", "-ld", action="store_true", default=False, help="Use light dataset")
    
    args = parser.parse_args()
    
    if args.use_wandb:
        import wandb
        wandb.login()
    
    configs_to_run = []
    
    assert(args.model_type.lower() in ["mlp", "lstm", "rf"]), f"Invalid model type: {args.model_type.lower()} expected one of mlp, lstm, rf"

    if os.path.isfile(args.hp_config_file) and args.hp_config_file.endswith(".yaml"):
        configs_to_run = [args.hp_config_file]
    else:
        raise ValueError(f"Invalid hyperparameters configuration file: {args.hp_config_file} (expected a YAML file)")
    
    prInfo(f"Running single experiment with hyperparameters configuration file: {configs_to_run[0]}")
    args.hp_config_file = configs_to_run[0]
    train_model(args, model_type=args.model_type)
    
