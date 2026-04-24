# Import standard libraries
import argparse
import os
# Define the here variable to be the directory of the current file
here = os.path.dirname(os.path.abspath(__file__))
from tqdm import tqdm
import joblib
import pandas as pd

# Display libraries
import cv2

# Import torch libraries
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

# Import random library
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve

# Import the datasets classes
# when using with more freedom for visualizing
from datasets.HUIDatasetInteractive import HUIInteract360Interactive
# when using strictly as in training
from datasets.HUIDataset import HUIInteract360

# Import the predictors classes
from predictors.mlp import MLPInteractionPredictor
from predictors.lstm import LSTMInteractionPredictor

# Import custom utils
from datasets.hui_norm_values import HUI_NORMALIZATION_VALUES
from utils.print_utils import *
from utils.other_utils import write_dic_to_yaml_file
from utils.visualize_utils import *

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
    


def display_input_tensor(input_tensor, metadata, label, proba_sequence, dataset, args):
    
    # Load background images for display
    bg_images_path = {recording_name: os.path.join(here, "datasets", "hf_data", "background_images", recording_name+".jpg") for recording_name in RECORDINGS_LIST}
    bg_images = {recording_name: cv2.imread(bg_image_path) for recording_name, bg_image_path in bg_images_path.items() if os.path.exists(bg_image_path)}

    unique_track_identifier = metadata["unique_track_identifier"][0]
    prInfo(f"Displaying track: {unique_track_identifier}") # e.g. rosbag2_2025_07_07-15_33_32_0029_1
    recording_name = unique_track_identifier.split("_")[0:6]
    recording_name = "_".join(recording_name)
    if recording_name in bg_images:
        background_image = bg_images[recording_name]
    else:
        background_image = np.zeros((1920, 3840, 3), dtype=np.uint8)
        
    data_columns = dataset.data_columns_in_dataset
    input_df = pd.DataFrame(input_tensor[0,:,:].cpu().numpy(), columns=data_columns)
        
    vitpose_array_in_box_coordinates = np.zeros((input_tensor.shape[1], 17, 3)) # T,17,3
    vitpose_array_in_image_coordinates = np.zeros((input_tensor.shape[1], 17, 3)) # T,17,3
    
    image_size = (background_image.shape[1], background_image.shape[0])
    
    # Extract metadata once (same for all keypoints)
    xmin_meta = metadata["xmin_meta"].cpu().numpy()[0,...]
    xmax_meta = metadata["xmax_meta"].cpu().numpy()[0,...]
    ymin_meta = metadata["ymin_meta"].cpu().numpy()[0,...]
    ymax_meta = metadata["ymax_meta"].cpu().numpy()[0,...]
    
    for kpt_idx, vitpose_kpt_name in enumerate(VITPOSE_KEYPOINTS_NAMES):
        
        if f"vitpose_{vitpose_kpt_name}_x" not in input_df.columns:
            continue
        
        kpt_x_standardize_mean = HUI_NORMALIZATION_VALUES[f"vitpose_{vitpose_kpt_name}_x_box_norm"]["mean"]
        kpt_x_standardize_std = HUI_NORMALIZATION_VALUES[f"vitpose_{vitpose_kpt_name}_x_box_norm"]["std"]
        kpt_y_standardize_mean = HUI_NORMALIZATION_VALUES[f"vitpose_{vitpose_kpt_name}_y_box_norm"]["mean"]
        kpt_y_standardize_std = HUI_NORMALIZATION_VALUES[f"vitpose_{vitpose_kpt_name}_y_box_norm"]["std"]
        
        # now destandardize the x and y values
        kpt_x_destandardized = input_df[f"vitpose_{vitpose_kpt_name}_x"].values * kpt_x_standardize_std + kpt_x_standardize_mean
        kpt_y_destandardized = input_df[f"vitpose_{vitpose_kpt_name}_y"].values * kpt_y_standardize_std + kpt_y_standardize_mean
        
        # now store the standardized x and y values
        vitpose_array_in_box_coordinates[:, kpt_idx, 0] = kpt_x_destandardized
        vitpose_array_in_box_coordinates[:, kpt_idx, 1] = kpt_y_destandardized
        vitpose_array_in_box_coordinates[:, kpt_idx, 2] = input_df[f"vitpose_{vitpose_kpt_name}_score"].values # this one was not standardized
        
        kpt_x_destandardized_in_image_coordinates = kpt_x_destandardized * (xmax_meta - xmin_meta) + xmin_meta
        kpt_y_destandardized_in_image_coordinates = kpt_y_destandardized * (ymax_meta - ymin_meta) + ymin_meta
        
        # now store the standardized x and y values
        vitpose_array_in_image_coordinates[:, kpt_idx, 0] = kpt_x_destandardized_in_image_coordinates
        vitpose_array_in_image_coordinates[:, kpt_idx, 1] = kpt_y_destandardized_in_image_coordinates
        vitpose_array_in_image_coordinates[:, kpt_idx, 2] = input_df[f"vitpose_{vitpose_kpt_name}_score"].values # this one was not standardized
    
    # Draw skeleton on full-size image
    image_size = (image_size[0] // 4, image_size[1] // 4)
    
    image = cv2.resize(background_image.copy(), image_size, interpolation=cv2.INTER_LINEAR)
    original_image = image.copy()
    
    transparent_image = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)
    transparent_image[:,:,3] = 0
    
    original_transparent_image = transparent_image.copy()
    
    if "albee" in unique_track_identifier or "astor" in unique_track_identifier:
        if np.any(vitpose_array_in_image_coordinates[:,:,0] > 3840):
            # for wrapped detections recenter them
            vitpose_array_in_image_coordinates[:,:,0] -= 1920
            xmin_meta -= 1920
            xmax_meta -= 1920
            
    vitpose_array_in_image_coordinates[:,:,0] = vitpose_array_in_image_coordinates[:,:,0] // 4
    vitpose_array_in_image_coordinates[:,:,1] = vitpose_array_in_image_coordinates[:,:,1] // 4
    
    boxes_xmin = xmin_meta.copy() // 4
    boxes_xmax = xmax_meta.copy() // 4
    boxes_ymin = ymin_meta.copy() // 4
    boxes_ymax = ymax_meta.copy() // 4
    
    # Draw all frames on the same image
    mini_vid = [] # just this one
    
    num_frames = vitpose_array_in_image_coordinates.shape[0]
    for frame_idx in range(num_frames):
        
        time_to_interaction = metadata["time_to_interaction_by_frame"][0][frame_idx].item()
        start_at_frame = metadata["input_index"].item()
        current_frame = start_at_frame + frame_idx
        index_in_recording = metadata["image_indexes_track"][frame_idx].item()
        
        overlay = cv2.addWeighted(original_image.copy(), args.overlay_weight, image, 1 - args.overlay_weight, 0)
        transparent_overlay = cv2.addWeighted(original_transparent_image.copy(), args.overlay_weight, transparent_image, 1 - args.overlay_weight, 0)
        
        box_xmin = boxes_xmin[frame_idx]
        box_xmax = boxes_xmax[frame_idx]
        box_ymin = boxes_ymin[frame_idx]
        box_ymax = boxes_ymax[frame_idx]
                
        # Draw skeleton connections first (so keypoints appear on top)
        for connection in VITPOSE_SKELETON:
            kpt1_idx = connection[0]
            kpt2_idx = connection[1]
            
            # Get keypoint coordinates
            kpt1_x = int(vitpose_array_in_image_coordinates[frame_idx, kpt1_idx, 0])
            kpt1_y = int(vitpose_array_in_image_coordinates[frame_idx, kpt1_idx, 1])
            kpt1_score = vitpose_array_in_image_coordinates[frame_idx, kpt1_idx, 2]
            
            kpt2_x = int(vitpose_array_in_image_coordinates[frame_idx, kpt2_idx, 0])
            kpt2_y = int(vitpose_array_in_image_coordinates[frame_idx, kpt2_idx, 1])
            kpt2_score = vitpose_array_in_image_coordinates[frame_idx, kpt2_idx, 2]
            
            # Only draw if both keypoints have valid scores (threshold > 0.1)
            if kpt1_score > 0.1 and kpt2_score > 0.1:
                # Check if coordinates are within image bounds
                if (0 <= kpt1_x < image_size[0] and 0 <= kpt1_y < image_size[1] and
                    0 <= kpt2_x < image_size[0] and 0 <= kpt2_y < image_size[1]):
                    # Use white/gray for skeleton connections, with slight variation for frame progression
                    alpha = 0.3 + 0.7 * (frame_idx + 1) / num_frames
                    connection_color = (int(200 * alpha), int(200 * alpha), int(200 * alpha))
                    cv2.line(overlay, (kpt1_x, kpt1_y), (kpt2_x, kpt2_y), connection_color, 2)
                    cv2.line(transparent_overlay, (kpt1_x, kpt1_y), (kpt2_x, kpt2_y), (connection_color[0], connection_color[1], connection_color[2], 255), 2)
        
        # Draw keypoints with their predefined colors
        for kpt_idx in range(17):
            kpt_x = int(vitpose_array_in_image_coordinates[frame_idx, kpt_idx, 0])
            kpt_y = int(vitpose_array_in_image_coordinates[frame_idx, kpt_idx, 1])
            kpt_score = vitpose_array_in_image_coordinates[frame_idx, kpt_idx, 2]
            
            # Only draw if keypoint has valid score
            if kpt_score > 0.1:
                # Check if coordinates are within image bounds
                if 0 <= kpt_x < image_size[0] and 0 <= kpt_y < image_size[1]:
                    # Use the predefined color for this keypoint type
                    kpt_color = VITPOSE_COLORS[kpt_idx]
                    # Make later frames slightly brighter to show progression
                    frame_alpha = 0.7 + 0.3 * (frame_idx + 1) / num_frames
                    adjusted_color = tuple(
                        min(255, int(c * frame_alpha)) 
                        for c in kpt_color
                    )
                    cv2.circle(overlay, (kpt_x, kpt_y), 6, adjusted_color, -1)
                    cv2.circle(overlay, (kpt_x, kpt_y), 7, (255, 255, 255), 1)
                    cv2.circle(transparent_overlay, (kpt_x, kpt_y), 6, (adjusted_color[0], adjusted_color[1], adjusted_color[2], 255), -1)
                    cv2.circle(transparent_overlay, (kpt_x, kpt_y), 7, (255, 255, 255, 255), 1)

        # Update the image
        image = overlay
        transparent_image = transparent_overlay
        
        image_show = image.copy() # for things that dont stay
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        font_thickness = 2
        text_color = (255, 255, 255) # White
        outline_color = (0, 0, 0)  # Black for outline
                
        # Add text information at the top center
        probability = proba_sequence[frame_idx]
        
        # if proba > -1 also write it on top of the box
        if probability > -1:
            cv2.putText(image_show, f"{probability*100:.0f}%", (int(box_xmin), int(box_ymin)), font, font_scale, text_color, font_thickness)
        
        proba_str = f"{probability:.3f}" if probability != -1 else "WAIT"
        if len(proba_sequence) == 30:
            # exact one segment prediction
            obs_window_str = f"{start_at_frame-1} - {start_at_frame+30-1}"            
        else:
            obs_window_str = f"{index_in_recording-30} - {index_in_recording}" if index_in_recording-30 >= start_at_frame else f"WAIT"
            
        prediction = 1 if probability > 0.5 else 0
        text_lines = [
            f"TrackID: {metadata['unique_track_identifier'][0].replace('rosbag2_2025_', '')} | Prob: {proba_str} | Ground Truth: {int(label)}",
            f"Time to interaction: {time_to_interaction:.2f}s | Frame: {index_in_recording} | Obs. window : {obs_window_str}",
        ]

        # Calculate text size and position for centering
        line_height = 25
        total_height = len(text_lines) * line_height
        start_y = 50
        
        for i, text in enumerate(text_lines):
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Calculate x position for centering
            x = (image_size[0] - text_width) // 2
            y = start_y + i * line_height
            
            # Draw outline (black) for better visibility
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx != 0 or dy != 0:
                        cv2.putText(image_show, text, (x + dx, y + dy), font, font_scale, outline_color, font_thickness + 2)
            
            # Draw main text
            cv2.putText(image_show, text, (x, y), font, font_scale, text_color, font_thickness)
        
        # Create separate chart image to concatenate below
        chart_height = 150
        chart_margin = 20
        chart_image = np.ones((chart_height, image_size[0], 3), dtype=np.uint8) * 255  # White background
        chart_top = chart_margin
        chart_bottom = chart_height - chart_margin
        chart_left = chart_margin
        chart_right = image_size[0] - chart_margin
        chart_width = chart_right - chart_left
        chart_plot_height = chart_bottom - chart_top
        
        # Draw grid lines and labels (only 0.0 and 1.0, no 0.5 threshold)
        grid_values = [0.0, 1.0]
        grid_colors = [(200, 200, 200), (200, 200, 200)]  # Gray for grid
        
        for val, color in zip(grid_values, grid_colors):
            y_pos = int(chart_bottom - val * chart_plot_height)
            label_text = f"{val:.1f}"
            (label_w, label_h), _ = cv2.getTextSize(label_text, font, 0.5, 1)
        
        # Draw probability curve progressively
        valid_probas = [p for p in proba_sequence[:frame_idx+1] if p > -1]
        if len(valid_probas) > 0:
            # Calculate x positions (equally spaced across chart width)
            num_valid = len(valid_probas)
            points = []
            for i, prob in enumerate(valid_probas):
                if num_valid > 1:
                    x = int(chart_left + (i / (num_valid - 1)) * chart_width)
                else:
                    x = int(chart_left + chart_width // 2)  # Center if only one point
                y = int(chart_bottom - prob * chart_plot_height)
                points.append((x, y))
            
            if num_valid > 1:
                # Draw line only (no filled area)
                for i in range(len(points) - 1):
                    pt1 = points[i]
                    pt2 = points[i + 1]
                    # Draw line
                    cv2.line(chart_image, pt1, pt2, (0, 150, 255), 3)  # Orange/blue color
            
            # Draw points on curve with progressive coloring based on probability value
            for i, (x, y) in enumerate(points):
                prob = valid_probas[i]
                # Map probability [0,1] -> color gradient: (0,100,255) (orange) to (0,200,0) (green)
                # Linear interpolation
                orange = np.array([0, 0, 255])
                green  = np.array([0, 255,   0])
                point_color = (orange * (1.0 - prob) + green * prob).astype(int)
                point_color = tuple(int(c) for c in point_color)
                cv2.circle(chart_image, (x, y), 4, point_color, -1)
                cv2.circle(chart_image, (x, y), 5, (255, 255, 255), 1)
            
            # Draw current point more prominently using same coloring logic
            if len(points) > 0:
                last_x, last_y = points[-1]
                last_prob = valid_probas[-1]
                orange = np.array([0, 0, 255])
                green  = np.array([0, 255,   0])
                current_color = (orange * (1.0 - last_prob) + green * last_prob).astype(int)
                current_color = tuple(int(c) for c in current_color)
                cv2.circle(chart_image, (last_x, last_y), 6, current_color, -1)
                cv2.circle(chart_image, (last_x, last_y), 7, (255, 255, 255), 2)
        
        # Draw vertical black bar at first interaction time
        # Check all frames up to current frame to find where interaction occurs
        interaction_frame_idx = None
        
        # First try time_to_first_interaction if available per frame
        if "time_to_first_interaction" in metadata:
            time_to_first_interaction = metadata["time_to_first_interaction"]
            if isinstance(time_to_first_interaction, torch.Tensor) and time_to_first_interaction.dim() > 0:
                # It's per-frame, check all frames
                for check_frame_idx in range(frame_idx + 1):
                    if check_frame_idx < len(time_to_first_interaction):
                        ttf = time_to_first_interaction[check_frame_idx].item()
                        if abs(ttf) < 0.1:
                            interaction_frame_idx = check_frame_idx
                            break
        
        # Fallback to time_to_interaction_by_frame if not found yet
        if interaction_frame_idx is None and "time_to_interaction_by_frame" in metadata:
            # Check all frames to find where interaction occurs
            for check_frame_idx in range(frame_idx + 1):
                if check_frame_idx < len(metadata["time_to_interaction_by_frame"][0]):
                    ttf_frame = metadata["time_to_interaction_by_frame"][0][check_frame_idx].item()
                    if abs(ttf_frame) < 0.1:
                        interaction_frame_idx = check_frame_idx
                        break
        
        # If interaction found, draw vertical black bar
        if interaction_frame_idx is not None:
            # Map the frame index to the chart x position
            # We need to find which position in valid_probas corresponds to interaction_frame_idx
            valid_frame_indices = [i for i in range(frame_idx + 1) if proba_sequence[i] > -1]
            if interaction_frame_idx in valid_frame_indices:
                valid_idx = valid_frame_indices.index(interaction_frame_idx)
                num_valid = len(valid_frame_indices)
                if num_valid > 1:
                    x_pos = int(chart_left + (valid_idx / (num_valid - 1)) * chart_width)
                else:
                    x_pos = int(chart_left + chart_width // 2)
                cv2.line(chart_image, (x_pos, chart_top), (x_pos, chart_bottom), (0, 0, 0), 2)

        # Concatenate chart below the main image
        image_show = np.vstack([image_show, chart_image])
        
        if args.save_video:
            mini_vid.append(image_show)
        

        cv2.imshow("Skeleton Visualization", image_show)
        if args.save_video:
            waitkey = 20
        else:
            waitkey = int(1000/15) if frame_idx < num_frames - 1 else 0
            
        key = cv2.waitKey(waitkey)
        if key == 27: # ESC key
            exit(0)
            
    if args.save_video:
        track_video_prediction_path = f"./track_videos_predictions/{unique_track_identifier}.mp4"
        if not os.path.exists("./track_videos_predictions"):
            os.makedirs("./track_videos_predictions")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(track_video_prediction_path, fourcc, 15, (image_show.shape[1], image_show.shape[0]))
        for frame in mini_vid:
            out.write(frame)
        out.release()
        prSuccess(f"Track video prediction saved to: {track_video_prediction_path}")
        
    cv2.destroyAllWindows()

    return 


def evaluate_without_display(model, dataloader, device):
    """Evaluate the model on a dataset."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    timings = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch_idx, (input_tensor, label, metadata) in enumerate(progress_bar):
                        
            # Move to device
            input_tensor = input_tensor.to(device)
            label = label.to(device).float()
            
            # Forward pass
            logits = model(input_tensor)

            # Collect predictions and probabilities
            probabilities = torch.sigmoid(logits[:, 0])
            predictions = probabilities > 0.5
            
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(label.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Processed': f'{batch_idx+1}/{len(dataloader)}'
            })
            
    # Calculate metrics
    if np.all([p == all_predictions[0] for p in all_predictions]):
        prWarning(f"[Evaluate] All predictions are the same : {all_predictions[0]}")
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        auc = 0.0
    else:
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        auc = roc_auc_score(all_labels, all_probabilities)
        
    return accuracy, precision, recall, f1, auc, all_labels, all_probabilities

def evaluate_with_display(model, dataloader, device, args=None):
    """Evaluate the model on a dataset."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    timings = []
    
    last_label = -1
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch_idx, (input_tensor, label, metadata) in enumerate(progress_bar):
            
            
            # Move to device
            input_tensor = input_tensor.to(device)            
            
            if input_tensor.shape[0] != 1:
                raise ValueError(f"Input tensor shape is not 1: {input_tensor.shape}. We only support batch size 1 for now.")

            
            if input_tensor.shape[1] != args.original_input_length_in_frames:
                                
                proba_display = [-1] * args.original_input_length_in_frames # to start with then fill with the probabilities
                # perform multiple inferences and slide along the input !
                for i in range(input_tensor.shape[1] - args.original_input_length_in_frames):
                    # print(f"iter {i}, sliding from {i} to {i+args.original_input_length_in_frames}")
                    input_tensor_slide = input_tensor[:, i:i+args.original_input_length_in_frames]
                    logits = model(input_tensor_slide)
                    probabilities = torch.sigmoid(logits[:, 0])
                    predictions = probabilities > 0.5
                    probability = probabilities.item()
                    proba_display.append(probability)
            else:   
                logits = model(input_tensor)
                probabilities = torch.sigmoid(logits[:, 0])
                predictions = probabilities > 0.5
                probability = probabilities.item()
                proba_display = [probability] * args.original_input_length_in_frames
            
            
            # print(proba_display, len(proba_display))
            
            label = label.to(device).float()

            if (args.show_select == "positives" and label.item() == 1) or (args.show_select == "negatives" and label.item() == 0):
                # Display input tensor
                display_input_tensor(input_tensor, metadata, label.item(), proba_display, dataloader.dataset, args)
            elif args.show_select == "balanced" and (label.item() != last_label or last_label == -1):
                # Display input tensor
                display_input_tensor(input_tensor, metadata, label.item(), proba_display, dataloader.dataset, args)
                last_label = label.item()
            elif args.show_select == "all":
                # Display input tensor
                display_input_tensor(input_tensor, metadata, label.item(), proba_display, dataloader.dataset, args)
            else:
                pass
            
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(label.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Processed': f'{batch_idx+1}/{len(dataloader)}'
            })
            
    # Calculate metrics
    if np.all([p == all_predictions[0] for p in all_predictions]):
        prWarning(f"[Evaluate] All predictions are the same : {all_predictions[0]}")
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        auc = 0.0
    else:
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        auc = roc_auc_score(all_labels, all_probabilities)
        
    return accuracy, precision, recall, f1, auc, all_labels, all_probabilities

def evaluate_rf(model, val_X, val_y, device):
    """Evaluate the RF model on a dataset."""

    predictions = model.predict(val_X)
    probabilities = model.predict_proba(val_X)[:, 1]
    predictions = (probabilities > 0.5).astype(np.int32)
    
    if np.all([p == predictions[0] for p in predictions]):
        prWarning(f"[Evaluate] All predictions are the same : {predictions[0]}")
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        auc = 0.0
    else:        
        accuracy = accuracy_score(val_y, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(val_y, predictions, average='binary')
        auc = roc_auc_score(val_y, probabilities)
        
    return accuracy, precision, recall, f1, auc, val_y, probabilities

def main(args, model_path):
    
    # Check if model file exists
    if not os.path.exists(model_path):
        prError(f"Model file not found: {model_path}")
        exit(1)
    
    prInfo(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract hyperparameters
    if 'hyperparameters' not in checkpoint:
        prError("Checkpoint does not contain 'hyperparameters' key. Cannot recreate dataloader.")
        exit(1)
    
    config = checkpoint['hyperparameters']
    prInfo("Loaded hyperparameters from checkpoint")
    
    prInfo(f"Model type: {config['force_model_type']}") # rf, mlp, lstm
    prInfo(f"Validation tracks filename: {config['val_tracks_filename']}")
    prInfo(f"Cross evaluation type: {config['cross_eval_type']}")

    # Set device
    if args.device == "auto":
        if "rf" in model_path:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    prInfo(f"Using device: {device}")
    
    # Create validation dataset using the saved hyperparameters
    prInfo("Creating validation dataset...")
    
    # Get validation tracks
    if config["val_tracks_filename"] != "all":
        tracks_file_path = os.path.join(here, "utils", "tests", config["val_tracks_filename"])
        if not os.path.exists(tracks_file_path):
            prError(f"Validation tracks file not found: {tracks_file_path}")
            exit(1)
        val_tracks = [line.strip() for line in open(tracks_file_path)]
    else:
        val_tracks = "all"
    
    # for reference later if not loaded with the right ones (when displaying we may take some liberty with the input length)
    args.original_input_length_in_frames = config["input_length_in_frames"] // config["subsample_frames"]
    prInfo(f"Original sequence length (after subsampling): {args.original_input_length_in_frames}")
    
    if not args.custom_hyperparameters_in_dataset:
        # Create validation dataset with the exact same hyperparameters as the ones used in the training script
        val_dataset = HUIInteract360(
            include_recordings=config["include_recordings_val"],
            include_tracks=val_tracks, 
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
            dataset_revision=config.get("dataset_revision", "3c8a342548534b6b92d32b0099e266962facdf45"),
        )
            
    else:
        # Create validation dataset with whatever we want for the display
        include_recordings = config["include_recordings_val"]
        if args.use_custom_recording:
            include_recordings = [args.use_custom_recording]
            
        # As it is HUIInteract360Interactive will create sequences of length 90 frames centered on the onset 
        # (either biggest mask size or interaction and will perform inference with a sliding windows 
        # of the length specified in the checkpoint, typically 30 frames)
        val_dataset = HUIInteract360Interactive(
            include_recordings=include_recordings,
            include_tracks=val_tracks, 
            include_columns=config["include_columns"],
            positive_cutoff=0, #config["positive_cutoff_val"],    
            interaction_cutoff=0, #config["interaction_cutoff_val"],
            fixed_input_length=config["fixed_input_length"],
            input_length_in_frames=args.show_length_frames, #config["input_length_in_frames"],
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
            force_positive_samples=False, #config["force_positive_samples"],
            ignore_negative_tracks_after_biggest_mask_size=False, #config["ignore_negative_tracks_after_biggest_mask_size_val"],
            force_aligment_with_biggest_mask_size=True, #config["force_aligment_with_biggest_mask_size_val"],
            verbose=False,
            center_on_onset=True,
            dataset_revision=config.get("dataset_revision", "3c8a342548534b6b92d32b0099e266962facdf45"),
        )
    
    # Print the number of positive and negative samples in the dataset
    val_positives = val_dataset.total_positives
    val_negatives = val_dataset.total_negatives
    prInfo(f"Validation positives: {val_positives}")
    prInfo(f"Validation negatives: {val_negatives}")
    prInfo(f"Validation samples: {len(val_dataset)}")
    
    
    # Get input dimensions
    input_dim = len(val_dataset.data_columns_in_dataset)
    sequence_length = val_dataset.input_length_in_frames // val_dataset.subsample_frames
    
    prInfo(f"Input dimension: {input_dim}")
    prInfo(f"Sequence length (after subsampling): {sequence_length}")
    
    
    # Create data loader
    # g = torch.Generator()
    # g.manual_seed(SEED)
    # if args.num_workers is None:
    #     num_workers = min(int(mp.cpu_count()-1), 32)
    #     prInfo(f"Using {num_workers} workers for dataloader (CPU count: {mp.cpu_count()})")
    # else:
    #     num_workers = args.num_workers
    #     prInfo(f"Using {num_workers} workers for dataloader")
    

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    if "rf" in model_path:
        # sklearn models
        val_X = []
        val_y = []
        for input_tensor, label, metadata in val_dataset:
            val_X.append(input_tensor.flatten())
            val_y.append(label)
        val_X = torch.stack(val_X).cpu().numpy()
        val_y = torch.stack(val_y).cpu().numpy()
        
        rf_model_path = model_path.replace("_model_", "_model_forest_").replace(".pth", ".joblib")
        model = joblib.load(rf_model_path)
        prInfo(f"RF model loaded from: {rf_model_path}")
        
    
    else:
        # torch models
        # batch_size = 1
        # val_dataloader = DataLoader(
        #     val_dataset, 
        #     batch_size=batch_size, 
        #     shuffle=False, 
        #     num_workers=num_workers, 
        #     worker_init_fn=seed_worker, 
        #     generator=g, 
        #     drop_last=False
        # )

        # Create model with the same architecture
        prInfo("Creating model...")
        
        if config['force_model_type'] == 'mlp':
            model = MLPInteractionPredictor(
                input_dim=input_dim,
                sequence_length=args.original_input_length_in_frames,
                hidden_dims=config["hidden_dims"],
                dropout=config["dropout"]
            ).to(device)
        elif config['force_model_type'] == 'lstm':
            model = LSTMInteractionPredictor(
                input_dim=input_dim,
                sequence_length=args.original_input_length_in_frames,
                hidden_dim=config["lstm_hidden_dim"],
                num_layers=config["lstm_num_layers"],
                dropout=config["lstm_dropout"],
                bidirectional=False
            ).to(device)
        else:
            raise ValueError(f"Model type {config['force_model_type']} not supported")
        
        prInfo(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
        
        # Load model weights
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Checkpoint does not contain 'model_state_dict' key. Cannot load model weights.")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        prSuccess("Model weights loaded successfully")
    
    
    # Print checkpoint info if available
    if 'val_auc' in checkpoint:
        expected_val_auc = checkpoint['val_auc']
        prInfo(f"Checkpoint validation AUC (expected): {expected_val_auc:.4f}")
    if 'val_f1' in checkpoint:
        expected_val_f1 = checkpoint['val_f1']
        # prInfo(f"Checkpoint validation F1 @0.5 (expected): {expected_val_f1:.4f}")
    if 'val_ap' in checkpoint:
        expected_val_ap = checkpoint['val_ap']
        prInfo(f"Checkpoint validation AP (expected): {expected_val_ap:.4f}")
    if 'epoch' in checkpoint:
        prInfo(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    # Run evaluation
    prInfo("Starting evaluation...")
    if "rf" in model_path:
        if args.display:
            raise NotImplementedError("RF models with display are not supported yet")
        else:
            accuracy, precision, recall, f1, auc, all_labels, all_probabilities = evaluate_rf(model, val_X, val_y, device)
    else:
        if args.display:
            accuracy, precision, recall, f1, auc, all_labels, all_probabilities = evaluate_with_display(model, val_dataloader, device, args)
        else:
            accuracy, precision, recall, f1, auc, all_labels, all_probabilities = evaluate_without_display(model, val_dataloader, device)
    
    # Print results
    prSuccess("\n" + "="*50)
    prSuccess("EVALUATION RESULTS")
    prSuccess("="*50)
    # prSuccess(f"Accuracy @0.5:  {accuracy:.4f}")
    # prSuccess(f"Precision @0.5: {precision:.4f}")
    # prSuccess(f"Recall @0.5:    {recall:.4f}")
    # prSuccess(f"F1 Score @0.5:  {f1:.4f}")
    prSuccess(f"AUC:       {auc:.4f}")
    prSuccess("="*50)
    
    if not args.custom_hyperparameters_in_dataset and args.light_dataset:
        # exact same dataset (without light dataset since we only have a partial dataset for size constraints it would not work)
        if np.abs(expected_val_auc - auc) > 0.001 and not args.custom_hyperparameters_in_dataset:
            prError(f"Expected validation AUC ({expected_val_auc:.4f}) and actual validation AUC ({auc:.4f}) differ by more than 0.001")
        elif not args.custom_hyperparameters_in_dataset:
            prSuccess("Validation AUC matches expected value !")        
    else:
        pass

    return all_labels, all_probabilities, config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on a trained model')
    parser.add_argument("--model_path", "-m", required=True, type=str, help="Path to the .pth model checkpoint file. Required.")
    parser.add_argument("--device", default="auto", type=str, help="Device to use (auto, cpu, cuda). Default: auto.")
    parser.add_argument("--num_workers", default=None, type=int, help="Number of workers for dataloader (default: auto)")
    parser.add_argument("--save_results", "-sr", action="store_true", default = False, help="Save the results. Default: False.")
    
    # display related args
    parser.add_argument("--display", "-d", action="store_true", default = False, help="Display the results. Default: False.")
    parser.add_argument("--overlay_weight", "-ow", default=0.1, type=float, help="Weight of the overlay (default: 0.9)")
    parser.add_argument("--show_select", "-ss", default="all", type=str, help="Show select (all, positives, negatives, balanced). Default: all.")
    parser.add_argument("--show_length_frames", "-sl", default=60, type=int, help="Show length in frames. Default: 60, minimum is 30.")
    
    # dataset related args
    parser.add_argument("--light_dataset", "-ld", action="store_true", default = False, help="Use light dataset (ie preprocessed, smaller, for supplementary material size limit). Default: False.")
    parser.add_argument("--custom_hyperparameters_in_dataset", "-chid", action="store_true", default = False, help="Custom hyperparameters in the dataset. Default: False.")
    parser.add_argument("--use_custom_recording", "-ucr", type=str, default = None, help="Use custom recording. Default: None.")

    # save video
    parser.add_argument("--save_video", "-sv", action="store_true", default = False, help="Save the video. Default: False.")

    args = parser.parse_args()
    
    assert args.show_length_frames >= 30, "Show length in frames must be at least 30"
    
    if args.use_custom_recording:
        assert args.custom_hyperparameters_in_dataset == True, "Cannot use custom recording without custom hyperparameters in the dataset"
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
    if os.path.isdir(args.model_path):
        # list all .pth in subdirectories, with depth max 1
        model_path_list = []
        subdirectories = [os.path.join(args.model_path, d) for d in os.listdir(args.model_path) if os.path.isdir(os.path.join(args.model_path, d))]
        for subdirectory in subdirectories:
            pth_files_in_subdirectory = [os.path.join(subdirectory, f) for f in os.listdir(subdirectory) if f.endswith(".pth")]
            if len(pth_files_in_subdirectory) > 1:
                prError(f"Found multiple .pth files in subdirectory: {subdirectory}")
                exit(1)
            elif len(pth_files_in_subdirectory) == 1:
                model_path_list.append(pth_files_in_subdirectory[0])
            else:
                prWarning(f"Found no .pth files in subdirectory: {subdirectory}")
                
        prInfo(f"Found {len(model_path_list)} model paths in subdirectories")
        prInfo(f"Model paths: {model_path_list}")
        
    else:
        model_path = args.model_path
        model_path_list = [model_path]
        
        
    for model_path in model_path_list:
        
        model_dir = os.path.dirname(model_path)
        all_labels, all_probabilities, config = main(args, model_path)    
        
        # Save all predictions and hyperparameters
        if not args.custom_hyperparameters_in_dataset and args.save_results:
            save_dir = os.path.join(model_dir, f"inference_results_{os.path.basename(model_path).split('.')[0]}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            else:
                prWarning(f"Save directory already exists: {save_dir}, do you want to overwrite it? (y/n)")
                overwrite = input()
                if overwrite == "y":
                    os.makedirs(save_dir, exist_ok=True)
                    prSuccess(f"Save directory overwritten: {save_dir}")
                else:
                    prError(f"Save directory not overwritten: {save_dir}, exiting...")
                    exit(1)
                    
            np.save(os.path.join(save_dir, "all_labels.npy"), np.array(all_labels))
            np.save(os.path.join(save_dir, "all_probabilities.npy"), np.array(all_probabilities))
            write_dic_to_yaml_file(config, os.path.join(save_dir, "hyperparameters.yaml"))
            
        elif args.custom_hyperparameters_in_dataset and args.save_results:
            prWarning("Skip saving all_predictions/all_probabilities.npy because we are not following original hyperparameters in the dataset")

