#!/usr/bin/env python3
"""
Qt-based visualizer for HUIInteract360 dataset
"""

import sys
import os
import argparse
import io

import numpy as np
import torch
import torch.nn as nn
import random
from scipy import ndimage
import cv2
import yaml
from datetime import datetime
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

from typing import List, Dict, Optional, Tuple, Any
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QPushButton, QCheckBox, QListWidget, QLabel, QSplitter,
    QGroupBox, QGridLayout, QAbstractItemView, QListWidgetItem,
    QScrollArea, QFrame, QMessageBox, QProgressBar, QStatusBar,
    QComboBox, QSpinBox, QTextEdit, QTabWidget, QDoubleSpinBox,
    QDialog, QDialogButtonBox, 
    QInputDialog, QFileDialog, QFormLayout, QButtonGroup, QRadioButton
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot, QRectF, QSize
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont, QKeySequence, 
    QShortcut, QBrush, QPainterPath, QPalette
)
from matplotlib.patches import Rectangle
from datasets.hui_norm_values import HUI_NORMALIZATION_VALUES
from datasets.HUIDatasetUtils import (
    input_tensor_to_format_by_channel, 
    input_tensor_to_format_by_channel_sapiens_without_face,
    keypoints17_to_coco18, 
    keypoints17_to_coco18_torch,
    coco2h36m, 
    crop_scale_torch, 
    crop_scale_torch_by_sample,
    coco2nwucla,
    sapiensnoface2nturgbd_nospine_mid
)
# Add current directory to path for imports
here = os.path.dirname(__file__)
sys.path.append(here)

from datasets.HUIDataset import HUIInteract360
from utils.data_utils import VITPOSE_KEYPOINTS_NAMES, METADATA_COLUMNS, FULL_DATA_COLUMNS
from utils.print_utils import prInfo, prWarning, prError, prSuccess, prDebug
from utils.visualize_utils import RECORDINGS_LIST, VITPOSE_COLORS, GOLIATH_KPTS_COLORS, GOLIATH_KEYPOINTS_NAMES, GOLIATH_SKELETON_INFO, UNIQUE_PLACES_RECORDINGS
from predictors.mlp import MLPInteractionPredictor
from predictors.lstm import LSTMInteractionPredictor
from predictors.STG_NF.model_pose import STG_NF
from predictors.STGCN.net.st_gcn import Model as STGCN
from predictors.SkateFormer.model.SkateFormer import SkateFormer
from tools.create_config_files import FEATURES_SET_D1, FEATURES_SET_D2, FEATURES_SET_D3, FEATURES_SET_D4, FEATURES_SET_D5, FEATURES_SET_D6, FEATURES_SET_D7

from utils.debug_utils import update_old_config_dict
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.eval_utils import get_best_threshold_f1

FEATURE_SET_DIC = {
    "D1 (mask)": FEATURES_SET_D1,
    "D2 (mask + box)": FEATURES_SET_D2,
    "D3 (mask + box + vitpose)": FEATURES_SET_D3,
    "D4 (mask + box + vitpose + sapiens)": FEATURES_SET_D4,
    "D5 (mask + box + vitpose + facial)": FEATURES_SET_D5,
    "D6 (mask + box + head/should)": FEATURES_SET_D6,
    "D7 (mask + vitpose)": FEATURES_SET_D7
}

class DatasetWorker(QThread):
    """Worker thread for dataset creation to avoid blocking the UI"""
    dataset_created = pyqtSignal(object, str)  # dataset, message
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def run(self):
        dataset = HUIInteract360(allow_download=False, **self.config)
        self.dataset_created.emit(dataset, f"Dataset created successfully! Length: {len(dataset)}")
        prSuccess(f"Dataset created successfully! Length: {len(dataset)}")
        prInfo("Visualized dataset statistics:")
        print(f"\tDataset positives (number of tracks): {dataset.total_positives_tracks}")
        print(f"\tDataset negatives (number of tracks): {dataset.total_negatives_tracks}")
        print(f"\tDataset possible positives (number of segments): {dataset.total_possible_positives_segments}")
        print(f"\tDataset possible negatives (number of segments): {dataset.total_possible_negatives_segments}")
        print(f"\tDataset used positive segments: {dataset.total_used_positive_segments}")
        print(f"\tDataset used negative segments: {dataset.total_used_negative_segments}")
        

class GraphWidget(QWidget):
    """Widget for displaying data plots with selectable columns"""
    
    def __init__(self, raw_data_path):
        super().__init__()
        self.dataset = None
        self.current_index = 0
        self.current_data = None
        self.dataset_stats = None
        self.proba_sequence = None  # Store probability sequence for current item
        self.current_metadata = None  # Store current metadata
        self.raw_data_path = raw_data_path  # Path to raw data directory
        self.current_images_tensor = None  # Store images tensor from dataset
        self.current_masks_tensor = None  # Store masks tensor from dataset
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Column selection controls
        controls_layout = QHBoxLayout()
        
        # Display mode selection (columns vs image)
        display_mode_layout = QVBoxLayout()
        display_mode_layout.addWidget(QLabel("Display Mode:"))
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["Select columns to plot", "Plot image (metadata)"])
        self.display_mode_combo.currentTextChanged.connect(self.update_plot)
        display_mode_layout.addWidget(self.display_mode_combo)
        controls_layout.addLayout(display_mode_layout)
        
        # Column selection list
        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.column_list.setMaximumHeight(150)
        self.column_list.itemSelectionChanged.connect(self.update_plot)
        
        controls_layout.addWidget(QLabel("Select columns to plot:"))
        controls_layout.addWidget(self.column_list)
        
        # Plot type selection
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Line Plot", "Scatter Plot", "Bar Plot"])
        self.plot_type_combo.currentTextChanged.connect(self.update_plot)
        controls_layout.addWidget(QLabel("Plot type:"))
        controls_layout.addWidget(self.plot_type_combo)
        
        # Statistics display options
        self.show_stats_checkbox = QCheckBox("Show Dataset Statistics")
        self.show_stats_checkbox.setChecked(False)
        self.show_stats_checkbox.toggled.connect(self.update_plot)
        controls_layout.addWidget(self.show_stats_checkbox)
        
        self.show_std_checkbox = QCheckBox("Show Standard Deviation")
        self.show_std_checkbox.setChecked(True)
        self.show_std_checkbox.toggled.connect(self.update_plot)
        controls_layout.addWidget(self.show_std_checkbox)
        
        layout.addLayout(controls_layout)
        
        # Split layout for main graph and skeleton view
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Main graph canvas
        self.figure = Figure(figsize=(3, 3))
        self.canvas = FigureCanvas(self.figure)
        splitter.addWidget(self.canvas)
        
        # Skeleton view panel
        skeleton_panel = QWidget()
        skeleton_layout = QVBoxLayout()
        
        # Skeleton controls
        skeleton_controls = QVBoxLayout()
        
        # Put skeleton checkboxes side by side
        skeleton_type_layout = QHBoxLayout()
        self.vitpose_skeleton_checkbox = QCheckBox("Show VitPose")
        self.vitpose_skeleton_checkbox.setChecked(True)
        self.vitpose_skeleton_checkbox.toggled.connect(self.update_skeleton)
        skeleton_type_layout.addWidget(self.vitpose_skeleton_checkbox)
        
        self.sapiens_skeleton_checkbox = QCheckBox("Show Sapiens")
        self.sapiens_skeleton_checkbox.setChecked(False)
        self.sapiens_skeleton_checkbox.toggled.connect(self.update_skeleton)
        skeleton_type_layout.addWidget(self.sapiens_skeleton_checkbox)
                
        self.free_scaleing_checkbox = QCheckBox("Free scale")
        self.free_scaleing_checkbox.setChecked(False)
        self.free_scaleing_checkbox.toggled.connect(self.update_skeleton)
        skeleton_type_layout.addWidget(self.free_scaleing_checkbox)
        
        self.show_bounding_box_checkbox = QCheckBox("Show BBox")
        self.show_bounding_box_checkbox.setChecked(True)
        self.show_bounding_box_checkbox.toggled.connect(self.update_skeleton)
        skeleton_type_layout.addWidget(self.show_bounding_box_checkbox)
        
        skeleton_controls.addLayout(skeleton_type_layout)

        # Timestamp slider
        self.timestamp_slider = QSlider(Qt.Orientation.Horizontal)
        self.timestamp_slider.setRange(0, 0)
        self.timestamp_slider.setValue(0)
        self.timestamp_slider.valueChanged.connect(self.update_skeleton)
        self.timestamp_slider.valueChanged.connect(self.update_plot)  # Also update plot when slider changes (for image mode)
        self.timestamp_slider.valueChanged.connect(self.update_image_display)  # Update image display when slider changes
        self.timestamp_label = QLabel("Frame: 0/0")
        skeleton_controls.addWidget(QLabel("Timestamp:"))
        skeleton_controls.addWidget(self.timestamp_slider)
        skeleton_controls.addWidget(self.timestamp_label)
        
        skeleton_layout.addLayout(skeleton_controls)
        
        # Skeleton canvas
        self.skeleton_figure = Figure(figsize=(3, 3))
        self.skeleton_canvas = FigureCanvas(self.skeleton_figure)
        skeleton_layout.addWidget(self.skeleton_canvas)
        
        skeleton_panel.setLayout(skeleton_layout)
        splitter.addWidget(skeleton_panel)
        
        # Image display panel
        image_panel = QWidget()
        image_layout = QVBoxLayout()
        
        # Image display label
        image_label = QLabel("Image Display:")
        image_layout.addWidget(image_label)
        
        # Image canvas
        self.image_figure = Figure(figsize=(3, 3))
        self.image_canvas = FigureCanvas(self.image_figure)
        image_layout.addWidget(self.image_canvas)
        
        image_panel.setLayout(image_layout)
        splitter.addWidget(image_panel)
        
        # Set splitter proportions (main graph, skeleton, image)
        splitter.setSizes([500, 250, 450])
        
        layout.addWidget(splitter)
        self.setLayout(layout)
    
    def set_dataset(self, dataset):
        """Set the dataset to visualize"""
        self.dataset = dataset
        self.current_index = 0
        if dataset is not None:
            self.populate_column_list()
            # self.update_current_data()
    
    def set_current_index(self, index):
        """Set the current item index"""
        self.current_index = index
        # self.update_current_data()
    
    def set_dataset_statistics(self, stats):
        """Set the dataset statistics for overlay display"""
        self.dataset_stats = stats
        self.update_plot()
    
    def populate_column_list(self):
        """Populate the column list with available data columns"""
        if self.dataset is None:
            return
        
        self.column_list.clear()
        data_columns = self.dataset.data_columns_in_dataset
        
        for col in data_columns:
            item = QListWidgetItem(col)
            self.column_list.addItem(item)
            if col == "mask_size":
                item.setSelected(True)
    
    def update_current_data(self, input_tensor, label, metadata, images_tensor=None, masks_tensor=None):
        """Update the current data from the dataset"""
        if self.dataset is None or len(self.dataset) == 0:
            self.current_data = None
            self.proba_sequence = None  # Clear proba_sequence when no data
            self.current_metadata = None
            self.current_images_tensor = None
            self.current_masks_tensor = None
            return
        
        # Store metadata and tensors
        self.current_metadata = metadata
        self.current_images_tensor = images_tensor
        self.current_masks_tensor = masks_tensor
        
        data_columns = self.dataset.data_columns_in_dataset
        
        # Convert tensor to numpy and create DataFrame
        input_data = input_tensor.numpy()
        self.current_data = pd.DataFrame(input_data, columns=data_columns)
        
        # Destandardize if enabled
        self.destandardize_data()
        
        # Add frame index as x-axis
        self.current_data['frame_index'] = range(len(self.current_data))
        
        # Update timestamp slider
        max_frames = len(self.current_data) - 1
        self.timestamp_slider.setRange(0, max_frames)
        self.timestamp_slider.setValue(0)
        # Update label with probability if available
        if self.proba_sequence is not None and 0 < len(self.proba_sequence):
            proba = self.proba_sequence[0]
            if proba > -1:
                proba_str = f"{proba:.3f}"
                proba_percent = f"{proba*100:.0f}%"
                self.timestamp_label.setText(f"Frame: 0/{max_frames} | Prob: {proba_str} ({proba_percent})")
            else:
                self.timestamp_label.setText(f"Frame: 0/{max_frames} | Prob: WAIT")
        else:
            self.timestamp_label.setText(f"Frame: 0/{max_frames}")
        
        self.update_plot()
        self.update_skeleton()
        self.update_image_display()
    
    def destandardize_data(self):
        """Destandardize the current data if destandardize is enabled"""
        # Get DataVisualizationWidget (parent widget)
        parent_widget = self.parent()
        if parent_widget is None or not hasattr(parent_widget, 'destandardize_checkbox'):
            self.destandardized = False
            return
    
        self.destandardized = True
        
        # Check if destandardize is enabled
        if not (parent_widget.destandardize_checkbox.isChecked() and 
                parent_widget.destandardize_checkbox.isEnabled()):
            return
        
        # Determine normalization suffix based on normalize_keypoints_in_box or normalize_keypoints_in_track from dataset
        normalize_in_box = getattr(self.dataset, 'normalize_keypoints_in_box', False)
        normalize_in_track = getattr(self.dataset, 'normalize_keypoints_in_track', "none")
        if normalize_in_box:
            norm_suffix = "_box_norm"
        elif normalize_in_track != "none":
            # norm_suffix = "_track_norm"
            norm_suffix = "_image_norm"
            # TODO : change this but for now we are using those values
        else:
            norm_suffix = "_image_norm"
        
        # Destandardize mask size
        if "mask_size" in self.current_data.columns and "mask_size_norm" in HUI_NORMALIZATION_VALUES:
            norm_values = HUI_NORMALIZATION_VALUES["mask_size_norm"]
            self.current_data["mask_size"] = (self.current_data["mask_size"] * norm_values["std"]) + norm_values["mean"]
        
        # Destandardize bounding box coordinates
        bbox_cols = ["xmin", "xmax", "ymin", "ymax"]
        for col in bbox_cols:
            if col in self.current_data.columns and col in HUI_NORMALIZATION_VALUES:
                norm_values = HUI_NORMALIZATION_VALUES[col]
                self.current_data[col] = (self.current_data[col] * norm_values["std"]) + norm_values["mean"]
        
        # Destandardize VitPose keypoints
        for kpt_name in VITPOSE_KEYPOINTS_NAMES:
            for coord in ["x", "y"]:
                col_name = f"vitpose_{kpt_name}_{coord}"
                norm_col_name = f"{col_name}{norm_suffix}"
                
                if col_name in self.current_data.columns and norm_col_name in HUI_NORMALIZATION_VALUES:
                    norm_values = HUI_NORMALIZATION_VALUES[norm_col_name]
                    self.current_data[col_name] = (self.current_data[col_name] * norm_values["std"]) + norm_values["mean"]
        
        # Destandardize Sapiens keypoints
        for kpt_name in GOLIATH_KEYPOINTS_NAMES:
            for coord in ["x", "y"]:
                col_name = f"sapiens_308_{kpt_name}_{coord}"
                norm_col_name = f"{col_name}{norm_suffix}"
                
                if col_name in self.current_data.columns and norm_col_name in HUI_NORMALIZATION_VALUES:
                    norm_values = HUI_NORMALIZATION_VALUES[norm_col_name]
                    self.current_data[col_name] = (self.current_data[col_name] * norm_values["std"]) + norm_values["mean"]
    
    def update_plot(self):
        """Update the plot with selected columns or image"""
        if self.current_data is None:
            self.figure.clear()
            self.canvas.draw()
            return
        
        # Check display mode
        display_mode = self.display_mode_combo.currentText()
        
        if display_mode == "Plot image (metadata)":
            # Display image instead of plotting columns
            self.plot_image()
            return
        
        # Original column plotting code
        # Get selected columns
        selected_columns = []
        for i in range(self.column_list.count()):
            if self.column_list.item(i).isSelected():
                selected_columns.append(self.column_list.item(i).text())
        
        if not selected_columns:
            self.figure.clear()
            self.canvas.draw()
            return
        
        # Clear the figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Get plot type
        plot_type = self.plot_type_combo.currentText()
        
        # Plot selected columns
        x_data = self.current_data['frame_index']
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_columns)))
        
        # Plot dataset statistics first (as background)
        if (self.dataset_stats is not None and 
            self.show_stats_checkbox.isChecked() and 
            plot_type == "Line Plot"):  # Only show stats for line plots
            
            for i, col in enumerate(selected_columns):
                if col in self.dataset_stats['columns']:
                    col_idx = self.dataset_stats['columns'].index(col)
                    
                    # Get statistics for this column
                    stats_means = self.dataset_stats['means'][:len(x_data), col_idx]
                    stats_stds = self.dataset_stats['stds'][:len(x_data), col_idx]
                    
                    # Plot mean line
                    ax.plot(x_data, stats_means, '--', color=colors[i], alpha=0.7, linewidth=1, 
                           label=f'{col} (dataset mean)')
                    
                    # Plot standard deviation band if enabled
                    if self.show_std_checkbox.isChecked():
                        ax.fill_between(x_data, 
                                      stats_means - stats_stds, 
                                      stats_means + stats_stds, 
                                      color=colors[i], alpha=0.2, 
                                      label=f'{col} (±1 std)')
        
        # Plot current data
        for i, col in enumerate(selected_columns):
            y_data = self.current_data[col]
            
            if plot_type == "Line Plot":
                ax.plot(x_data, y_data, label=col, color=colors[i], linewidth=2)
            elif plot_type == "Scatter Plot":
                ax.scatter(x_data, y_data, label=col, color=colors[i], alpha=0.7)
            elif plot_type == "Bar Plot":
                ax.bar(x_data, y_data, label=col, color=colors[i], alpha=0.7)
        
        # Customize the plot
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Value')
        ax.set_title(f'Data Visualization - Item {self.current_index + 1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adjust layout and refresh
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_image(self):
        """Plot the current image from metadata"""
        if self.current_metadata is None or self.raw_data_path is None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'No metadata or raw data path available', 
                   ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
            return
        
        # Get current frame index from timestamp slider
        frame_idx = self.timestamp_slider.value()
        
        # Extract metadata fields
        try:
            # Get unique_track_identifier and extract recording and episode from it
            # Format: RECORDINGNAME_EPISODE_TRACKID (e.g., "rosbag2_2025_07_07-12_38_45_001_1")
            unique_track_identifier = self.current_metadata["unique_track_identifier"]
            
            # Handle different metadata formats
            if isinstance(unique_track_identifier, torch.Tensor):
                if unique_track_identifier.dim() > 0 and len(unique_track_identifier) > 0:
                    unique_track_identifier = unique_track_identifier[0].item()
                else:
                    unique_track_identifier = unique_track_identifier.item()
            elif isinstance(unique_track_identifier, (list, np.ndarray)):
                if len(unique_track_identifier) > 0:
                    unique_track_identifier = unique_track_identifier[0]
                else:
                    unique_track_identifier = ""
            
            unique_track_identifier = str(unique_track_identifier)
            
            # Extract recording and episode from unique_track_identifier
            # Format: RECORDINGNAME_EPISODE_TRACKID
            # Example: "rosbag2_2025_07_07-15_33_32_0029_1" -> recording="rosbag2_2025_07_07-15_33_32", episode="0029"
            # Based on infer.py logic: recording is first 6 parts joined, episode is typically 4-digit number before track ID
            parts = unique_track_identifier.rsplit("_", 2)
            recording = parts[0]
            episode = parts[1]
            track_id = parts[2]
            
            image_indexes = self.current_metadata["image_indexes"] # list of integer

            # Convert to strings
            recording = str(recording)
            episode = str(episode)
            
            video_dir = os.path.join(self.raw_data_path, "video_mini", recording, "episodes", episode)
                        
            # Construct image path
            video_path = os.path.join(
                video_dir,
                "images_360_mini.mp4"
            )
            
            # Check if image exists
            if not os.path.exists(video_path):
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Image not found:\n{video_path}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                self.canvas.draw()
                return
            
            # Load and display image
            # img = Image.open(image_path)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, image_indexes[frame_idx])
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Error loading video:\n{video_path}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                self.canvas.draw()
                return
            
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.imshow(frame)
            ax.axis('off')
            ax.set_title(f'Frame {frame_idx}: {image_indexes[frame_idx]}')
            # Set axis limits to match image dimensions
            ax.set_xlim(0, frame.shape[1])
            ax.set_ylim(frame.shape[0], 0)  # Inverted y-axis to match image coordinates
                        
            # Check if shift was applied (do_recenter_interaction_zone was used)
            shift_applied = self.current_metadata["shift_applied"]
            interaction_zone_center_positions = self.current_metadata["interaction_zone_center_position"]
            xmin_meta = self.current_metadata["xmin_meta"]
            xmax_meta = self.current_metadata["xmax_meta"]
            ymin_meta = self.current_metadata["ymin_meta"]
            ymax_meta = self.current_metadata["ymax_meta"]
            image_width = self.current_metadata["image_size"][0]
            image_height = self.current_metadata["image_size"][1]
            
            # Convert tensors to numpy arrays if needed
            if xmin_meta is not None:
                if isinstance(xmin_meta, torch.Tensor):
                    xmin_meta = xmin_meta.cpu().numpy()
                elif isinstance(xmin_meta, list):
                    xmin_meta = np.array(xmin_meta)
            if xmax_meta is not None:
                if isinstance(xmax_meta, torch.Tensor):
                    xmax_meta = xmax_meta.cpu().numpy()
                elif isinstance(xmax_meta, list):
                    xmax_meta = np.array(xmax_meta)
            if ymin_meta is not None:
                if isinstance(ymin_meta, torch.Tensor):
                    ymin_meta = ymin_meta.cpu().numpy()
                elif isinstance(ymin_meta, list):
                    ymin_meta = np.array(ymin_meta)
            if ymax_meta is not None:
                if isinstance(ymax_meta, torch.Tensor):
                    ymax_meta = ymax_meta.cpu().numpy()
                elif isinstance(ymax_meta, list):
                    ymax_meta = np.array(ymax_meta)
            if interaction_zone_center_positions is not None:
                if isinstance(interaction_zone_center_positions, torch.Tensor):
                    interaction_zone_center_positions = interaction_zone_center_positions.cpu().numpy()
                elif isinstance(interaction_zone_center_positions, list):
                    interaction_zone_center_positions = np.array(interaction_zone_center_positions)
            
            has_shift = (shift_applied is not None and interaction_zone_center_positions is not None)
            
            # Draw original bounding box from metadata if shift was applied
            if has_shift:
                # Get original bounding box coordinates from metadata
                orig_xmin = xmin_meta[frame_idx]/image_width * frame.shape[1]
                orig_xmax = xmax_meta[frame_idx]/image_width * frame.shape[1]
                orig_ymin = ymin_meta[frame_idx]/image_height * frame.shape[0]
                orig_ymax = ymax_meta[frame_idx]/image_height * frame.shape[0]
                
                if not (np.isnan(orig_xmin) or np.isnan(orig_xmax) or np.isnan(orig_ymin) or np.isnan(orig_ymax)):
                    orig_bbox_width = orig_xmax - orig_xmin
                    orig_bbox_height = orig_ymax - orig_ymin
                    
                    # Draw original bounding box (green, solid, normal thickness)
                    orig_bbox_rect = Rectangle(
                        (orig_xmin, orig_ymin), 
                        orig_bbox_width, 
                        orig_bbox_height,
                        linewidth=3, 
                        edgecolor='green', 
                        facecolor='lightgreen', 
                        alpha=0.3,
                        linestyle='-'
                    )
                    ax.add_patch(orig_bbox_rect)
            
            # Draw bounding box if available in current_data (after destandardization/shifting)
            if self.current_data is not None and frame_idx < len(self.current_data):
                frame_data = self.current_data.iloc[frame_idx]
                
                if 'xmin' in frame_data and 'xmax' in frame_data and 'ymin' in frame_data and 'ymax' in frame_data:
                    xmin = frame_data['xmin'] * frame.shape[1]
                    xmax = frame_data['xmax'] * frame.shape[1]
                    ymin = frame_data['ymin'] * frame.shape[0]
                    ymax = frame_data['ymax'] * frame.shape[0]
                    
                    # Check if bounding box values are valid (not NaN)
                    if not (np.isnan(xmin) or np.isnan(xmax) or np.isnan(ymin) or np.isnan(ymax)):
                        # Calculate bounding box dimensions
                        bbox_width = xmax - xmin
                        bbox_height = ymax - ymin
                        
                        # Draw shifted bounding box (if shift was applied, use dashed and thinner)
                        if has_shift:
                            # Shifted box: thinner, dashed line
                            bbox_rect = Rectangle(
                                (xmin, ymin), 
                                bbox_width, 
                                bbox_height,
                                linewidth=1.5, 
                                edgecolor='green', 
                                facecolor='none', 
                                alpha=0.6,
                                linestyle='--'
                            )
                        else:
                            # Normal box: regular thickness, solid line
                            bbox_rect = Rectangle(
                                (xmin, ymin), 
                                bbox_width, 
                                bbox_height,
                                linewidth=3, 
                                edgecolor='green', 
                                facecolor='lightgreen', 
                                alpha=0.3,
                                linestyle='-'
                            )
                        ax.add_patch(bbox_rect)
            
            # Draw interaction zone center position icon at the bottom
            if interaction_zone_center_positions is not None and len(interaction_zone_center_positions) > frame_idx:
                iz_center_x = interaction_zone_center_positions[frame_idx]/image_width * frame.shape[1]
                if isinstance(iz_center_x, torch.Tensor):
                    iz_center_x = iz_center_x.item()
                if not np.isnan(iz_center_x) and iz_center_x >= 0:
                    # Convert to pixel coordinates
                    # interaction_zone_center_position is in pixel coordinates (not normalized)
                    if iz_center_x <= 1.0:
                        iz_center_x_px = iz_center_x * frame.shape[1]
                    else:
                        iz_center_x_px = iz_center_x
                    
                    # Draw small icon at bottom of image using Circle patch
                    icon_y = frame.shape[0] - 20  # 20 pixels from bottom
                    from matplotlib.patches import Circle
                    iz_circle = Circle((iz_center_x_px, icon_y), 8, 
                                      fill=True, facecolor='red', edgecolor='darkred', 
                                      linewidth=2, alpha=0.8, zorder=10)
                    ax.add_patch(iz_circle)
                    # Add small vertical line to show it's at the bottom
                    ax.plot([iz_center_x_px, iz_center_x_px], [icon_y, frame.shape[0]], 
                           'r-', linewidth=2, alpha=0.6, zorder=9)
            
            # Draw virtual center position icon (where interaction zone gets recentered to) at the bottom
            if has_shift:
                # Virtual center is at the center of the image
                virtual_center_x_px = frame.shape[1] / 2
                icon_y = frame.shape[0] - 20  # 20 pixels from bottom
                # Draw dashed circle using Circle patch with proper linestyle
                from matplotlib.patches import Circle
                virtual_circle = Circle((virtual_center_x_px, icon_y), 8, 
                                       fill=False, edgecolor='red', linewidth=2, 
                                       linestyle='--', alpha=0.8, zorder=10)
                ax.add_patch(virtual_circle)
                # Add small vertical dashed line to show it's at the bottom
                ax.plot([virtual_center_x_px, virtual_center_x_px], [icon_y, frame.shape[0]], 
                       'r--', linewidth=2, alpha=0.6, zorder=9, dashes=(5, 5))
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error loading image:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            self.canvas.draw()
    
    def update_skeleton(self):
        """Update the skeleton visualization"""
        if self.current_data is None:
            self.skeleton_figure.clear()
            self.skeleton_canvas.draw()
            return
        
        # Clear the skeleton figure
        self.skeleton_figure.clear()
        
        # Check if we should show skeletons and bounding box
        show_vitpose = self.vitpose_skeleton_checkbox.isChecked()
        show_sapiens = self.sapiens_skeleton_checkbox.isChecked()
        show_bounding_box = self.show_bounding_box_checkbox.isChecked()
        free_scaleing = self.free_scaleing_checkbox.isChecked()
                
        # Create subplots - left for skeleton, right for bounding box
        if show_bounding_box and (show_vitpose or show_sapiens):
            # Two subplots side by side
            ax_skeleton = self.skeleton_figure.add_subplot(121)
            ax_bbox = self.skeleton_figure.add_subplot(122)
        elif show_bounding_box:
            # Only bounding box
            ax_skeleton = None
            ax_bbox = self.skeleton_figure.add_subplot(111)
        else:
            # Only skeleton
            ax_skeleton = self.skeleton_figure.add_subplot(111)
            ax_bbox = None
        
        # Get current timestamp
        timestamp = self.timestamp_slider.value()
        # Update label with probability if available
        if self.proba_sequence is not None and timestamp < len(self.proba_sequence):
            proba = self.proba_sequence[timestamp]
            if proba > -1:
                proba_str = f"{proba:.3f}"
                proba_percent = f"{proba*100:.0f}%"
                self.timestamp_label.setText(f"Frame: {timestamp}/{len(self.current_data)-1} | Prob: {proba_str} ({proba_percent})")
            else:
                self.timestamp_label.setText(f"Frame: {timestamp}/{len(self.current_data)-1} | Prob: WAIT")
        else:
            self.timestamp_label.setText(f"Frame: {timestamp}/{len(self.current_data)-1}")
        

        if not show_vitpose and not show_sapiens and not show_bounding_box:
            # Use the main axis for the message
            main_ax = ax_skeleton if ax_skeleton is not None else ax_bbox
            if main_ax is not None:
                main_ax.text(0.5, 0.5, 'Select skeleton type or bounding box to display', 
                           ha='center', va='center', transform=main_ax.transAxes)
            self.skeleton_canvas.draw()
            return
        
        # Get image dimensions for coordinate scaling
        image_width, image_height = self.get_image_dimensions()
        is_normalized = self.is_data_normalized()
        
        # Define skeleton connections
        vitpose_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
            (5, 11), (6, 12), (11, 12),  # torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # legs
        ]
        
        # For Sapiens, we'll use a simplified skeleton with main body parts
        # (since it has 308 keypoints, we'll focus on the main body structure)
        # sapiens_main_keypoints = [
        #     'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        #     'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        #     'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        #     'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        # ]
        
        # sapiens_connections = [
        #     (0, 1), (0, 2), (1, 3), (2, 4),  # head
        #     (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        #     (5, 11), (6, 12), (11, 12),  # torso
        #     (11, 13), (13, 15), (12, 14), (14, 16)  # legs
        # ]
        
        # Plot bounding box
        if show_bounding_box and ax_bbox is not None:
            self.plot_bounding_box(ax_bbox, timestamp, is_normalized, image_width, image_height)
        
        # Plot VitPose skeleton
        if show_vitpose and ax_skeleton is not None:
            keypoints_colors = [(color[0]/255, color[1]/255, color[2]/255) for color in VITPOSE_COLORS]
            self.plot_skeleton(ax_skeleton, timestamp, 'vitpose', VITPOSE_KEYPOINTS_NAMES, 
                             vitpose_connections, 'red', keypoints_colors, 'VitPose', is_normalized, image_width, image_height)
        
        # Plot Sapiens skeleton
        if show_sapiens and ax_skeleton is not None:
            keypoints_colors = [(color[0]/255, color[1]/255, color[2]/255) for color in GOLIATH_KPTS_COLORS]
            sapiens_connections = [GOLIATH_SKELETON_INFO[i]['link'] for i in range(len(GOLIATH_SKELETON_INFO))]
            sapiens_main_keypoints = GOLIATH_KEYPOINTS_NAMES
            self.plot_skeleton(ax_skeleton, timestamp, 'sapiens_308', sapiens_main_keypoints, 
                             sapiens_connections, 'blue', keypoints_colors, 'Sapiens', is_normalized, image_width, image_height)
        
        # Set up the plot with dynamic limits for both axes
        axes_to_setup = []
        if ax_skeleton is not None:
            axes_to_setup.append(('Skeleton', ax_skeleton))
        if ax_bbox is not None:
            axes_to_setup.append(('Bounding Box', ax_bbox))
        
        for axis_name, ax in axes_to_setup:
            if free_scaleing:
                pass
                #ax.set_xlim(min_x, max_x)
                #ax.set_ylim(min_y, max_y)
            else:
                if self.is_standardized():
                    if self.destandardized:
                        if is_normalized:
                            ax.set_xlim(0, 1)
                            ax.set_ylim(0, 1)
                        else:
                            ax.set_xlim(0, image_width)
                            ax.set_ylim(0, image_height)
                    else:
                        pass
                else:
                    if is_normalized:
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                    else:
                        ax.set_xlim(0, image_width)
                        ax.set_ylim(0, image_height)
            
            ax.set_aspect('equal')
            ax.invert_yaxis()  # Invert Y axis to match image coordinates
            # ax.set_title(f'{axis_name} - Frame {timestamp} ({image_width}x{image_height})')
            ax.grid(True, alpha=0.3)
        
        self.skeleton_figure.tight_layout()
        self.skeleton_canvas.draw()
    
    def get_image_dimensions(self):
        """Get image dimensions from the current dataset item"""
        if self.dataset is None or self.current_index >= len(self.dataset):
            return 3840, 1920  # Default dimensions
        
        try:
            metadata = self.current_item_metadata
            if 'image_size' in metadata:
                width, height = metadata['image_size']
                return width, height
        except:
            pass
        
        return 3840, 1920  # Default fallback
    
    def is_data_normalized(self):
        """Check if the current data is normalized"""
        if self.dataset is None:
            return True  # Default to normalized
        
        # Check if normalize_in_image is enabled in the dataset
        return getattr(self.dataset, 'normalize_in_image', True)
    
    def is_standardized(self):
        """Check if the current data is standardized"""
        if self.dataset is None:
            return True  # Default to standardized
        
        standardize_attr = getattr(self.dataset, 'standardize_data', None)
        if standardize_attr == "all":
            return True
        else:
            return False
    
    def plot_skeleton(self, ax, timestamp, prefix, keypoint_names, connections, color, keypoints_colors, label, is_normalized, image_width, image_height):
        """Plot a skeleton for the given timestamp with dynamic coordinate handling"""
        if timestamp >= len(self.current_data):
            return
        
        # Get keypoint data for this timestamp
        frame_data = self.current_data.iloc[timestamp]
        
        # Extract keypoint coordinates
        keypoints = []
        valid_keypoints = []
        
        for i, name in enumerate(keypoint_names):
            x_col = f"{prefix}_{name}_x"
            y_col = f"{prefix}_{name}_y"
            score_col = f"{prefix}_{name}_score"
            
            if x_col in frame_data and y_col in frame_data and score_col in frame_data:
                x = frame_data[x_col]
                y = frame_data[y_col]
                score = frame_data[score_col]
                
                # Only plot keypoints with reasonable confidence
                if not np.isnan(x) and not np.isnan(y) and score > 0.3:
                    x_coord, y_coord = x, y
                    
                    keypoints.append((x_coord, y_coord))
                    valid_keypoints.append(i)
                else:
                    keypoints.append((np.nan, np.nan))
                    valid_keypoints.append(-1)
            else:
                keypoints.append((np.nan, np.nan))
                valid_keypoints.append(-1)
        
        # Plot keypoints with dynamic sizing
        valid_x = [kp[0] for i, kp in enumerate(keypoints) if valid_keypoints[i] != -1]
        valid_y = [kp[1] for i, kp in enumerate(keypoints) if valid_keypoints[i] != -1]
        colors = [keypoints_colors[i] for i in valid_keypoints if i != -1]
        
        if not "sapiens" in prefix:
            # Plot connections with dynamic line width
            line_width = max(1, min(4, (image_width + image_height) / 200))
            for connection in connections:
                start_idx, end_idx = connection
                if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                    valid_keypoints[start_idx] != -1 and valid_keypoints[end_idx] != -1):
                    
                    start_point = keypoints[start_idx]
                    end_point = keypoints[end_idx]
                    
                    if not (np.isnan(start_point[0]) or np.isnan(end_point[0])):
                        ax.plot([start_point[0], end_point[0]], 
                            [start_point[1], end_point[1]], 
                            color=color, linewidth=line_width, alpha=0.8)
    
        if valid_x and valid_y:
            # Dynamic point size based on image dimensions
            if "sapiens" in prefix:
                point_size = max(5, min(25, (image_width + image_height) / 100))
            else:
                point_size = max(20, min(100, (image_width + image_height) / 20))
                
            ax.scatter(valid_x, valid_y, c=colors, s=point_size, label=label, edgecolors='white', linewidth=1)
        
        # # Add legend only if we have valid keypoints
        # if valid_x and valid_y:
        #     ax.legend(loc='upper right', fontsize=8)
    
    def plot_bounding_box(self, ax, timestamp, is_normalized, image_width, image_height):
        """Plot the bounding box for the given timestamp in its dedicated axis"""
        if timestamp >= len(self.current_data):
            return
        
        # Get frame data
        frame_data = self.current_data.iloc[timestamp]
        
        # Extract bounding box coordinates
        if 'xmin' in frame_data and 'xmax' in frame_data and 'ymin' in frame_data and 'ymax' in frame_data:
            xmin = frame_data['xmin']
            xmax = frame_data['xmax']
            ymin = frame_data['ymin']
            ymax = frame_data['ymax']
            
            # Check if bounding box data is valid
            if not (np.isnan(xmin) or np.isnan(xmax) or np.isnan(ymin) or np.isnan(ymax)):
                # Calculate bounding box dimensions
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                
                # Create rectangle for bounding box with fill
                from matplotlib.patches import Rectangle
                bbox_rect = Rectangle((xmin, ymin), bbox_width, bbox_height, 
                                    linewidth=3, edgecolor='green', facecolor='lightgreen', 
                                    alpha=0.3, linestyle='-', label='Bounding Box')
                ax.add_patch(bbox_rect)
                
                # Add center point
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                ax.scatter(center_x, center_y, c='darkgreen', s=100, alpha=0.9, 
                          marker='+', linewidth=4, label='BBox Center')
                
                # Add corner points
                # corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                # for i, (cx, cy) in enumerate(corners):
                #     ax.scatter(cx, cy, c='green', s=60, alpha=0.8, 
                #               marker='o', edgecolors='darkgreen', linewidth=2)
                    # # Add corner labels
                    # corner_labels = ['TL', 'TR', 'BR', 'BL']  # Top-Left, Top-Right, Bottom-Right, Bottom-Left
                    # ax.text(cx, cy, corner_labels[i], fontsize=8, ha='center', va='center',
                    #        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                # Add detailed text annotation
                bbox_info = f'Size: {bbox_width:.2f} × {bbox_height:.2f}\nCenter: ({center_x:.2f}, {center_y:.2f})'
                ax.text(0.25, 0.95, bbox_info, transform=ax.transAxes, 
                       fontsize=10, color='darkgreen', alpha=0.9, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
                
                # Add legend
                # ax.legend(loc='upper right', fontsize=8)
    
    def update_image_display(self):
        """Update the image display with current frame and mask overlay"""
        self.image_figure.clear()
        ax = self.image_figure.add_subplot(111)
        
        # Get current frame index from timestamp slider
        frame_idx = self.timestamp_slider.value()
        
        # Check if images are available
        images_available = (self.current_images_tensor is not None and 
                           isinstance(self.current_images_tensor, (list, torch.Tensor)) and 
                           len(self.current_images_tensor) > 0)
        
        # Check if masks are available
        masks_available = (self.current_masks_tensor is not None and 
                          isinstance(self.current_masks_tensor, (list, torch.Tensor)) and 
                          len(self.current_masks_tensor) > 0)
        
        # If no images and no masks, show message
        if not images_available and not masks_available:
            ax.text(0.5, 0.5, 'No images or masks available.\nEnable "Return Images" or "Return Masks" in dataset settings.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            self.image_canvas.draw()
            return
        
        # Determine max frames from images or masks
        if images_available:
            max_frames = len(self.current_images_tensor)
            if frame_idx >= max_frames:
                ax.text(0.5, 0.5, f'Frame index {frame_idx} out of range.\nMax frames: {max_frames}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.axis('off')
                self.image_canvas.draw()
                return
        elif masks_available:
            max_frames = len(self.current_masks_tensor)
            if frame_idx >= max_frames:
                ax.text(0.5, 0.5, f'Frame index {frame_idx} out of range.\nMax frames: {max_frames}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.axis('off')
                self.image_canvas.draw()
                return
        
        # Get mask if available
        mask_np = None
        if masks_available and frame_idx < len(self.current_masks_tensor):
            mask = self.current_masks_tensor[frame_idx]  # Shape: (H, W)
            
            # Convert mask to numpy if it's a tensor
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            # Ensure mask is boolean
            if mask_np.dtype != bool:
                mask_np = mask_np.astype(bool)
        
        # If images are available, display them
        if images_available:
            # Get image at current frame_idx
            # images_tensor shape: (T, C, H, W) where T=time, C=channels, H=height, W=width
            image_tensor = self.current_images_tensor[frame_idx]  # Shape: (C, H, W)
            
            # Convert tensor to numpy array and handle different tensor formats
            if isinstance(image_tensor, torch.Tensor):
                # Convert from (C, H, W) to (H, W, C) for matplotlib
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = np.array(image_tensor)
                if image_np.shape[0] == 3:  # If (C, H, W), transpose to (H, W, C)
                    image_np = np.transpose(image_np, (1, 2, 0))
            
            # Handle uint8 format (0-255) vs float format (0-1)
            if image_np.dtype == np.uint8:
                image_np = image_np.astype(np.float32) / 255.0
            elif image_np.max() > 1.0:
                image_np = image_np.astype(np.float32) / 255.0
            
            # Clip values to [0, 1] range
            image_np = np.clip(image_np, 0, 1)
            
            # Display the image
            ax.imshow(image_np)
            ax.axis('off')
            ax.set_title(f'Frame {frame_idx}/{max_frames-1}')
            
            # Overlay mask if available
            if mask_np is not None:
                # Create colored mask overlay (semi-transparent green)
                colored_mask = np.zeros((*mask_np.shape, 4))  # RGBA
                colored_mask[mask_np] = [0, 1, 0, 0.4]  # Green with 40% opacity
                ax.imshow(colored_mask, interpolation='nearest')
                
        
        # If no images but masks are available, show mask on black background
        elif mask_np is not None:
            # Create black background with same dimensions as mask
            black_bg = np.zeros((*mask_np.shape, 3))  # RGB black image
            
            # Display black background
            ax.imshow(black_bg, interpolation='nearest')
            ax.axis('off')
            ax.set_title(f'Frame {frame_idx}/{max_frames-1} (Mask only)')
            
            # Display mask in white/colored
            # Create colored mask overlay (semi-transparent green)
            colored_mask = np.zeros((*mask_np.shape, 4))  # RGBA
            colored_mask[mask_np] = [0, 1, 0, 0.6]  # Green with 60% opacity (more visible on black)
            ax.imshow(colored_mask, interpolation='nearest')
            
            # Highlight mask border
            # Find edges of the mask
            mask_edges = mask_np.astype(float) - ndimage.binary_erosion(mask_np).astype(float)
            # Draw border in red
            border_mask = np.zeros((*mask_np.shape, 4))
            border_mask[mask_edges > 0] = [1, 0, 0, 0.9]  # Red border with 90% opacity
            ax.imshow(border_mask, interpolation='nearest')
        
        self.image_figure.tight_layout()
        self.image_canvas.draw()


class TrackFilterDialog(QDialog):
    """Dialog for entering track identifiers to filter"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Tracks by Identifier")
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)
        
        layout = QVBoxLayout()
        
        # Instructions label
        instructions = QLabel("Enter comma-separated unique_track_identifiers:")
        layout.addWidget(instructions)
        
        # Text input area
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("e.g., track1, track2, track3")
        layout.addWidget(self.text_input)
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_track_identifiers(self):
        """Get the list of track identifiers from the text input"""
        text = self.text_input.toPlainText().strip()
        if not text:
            return []
        # Split by comma and strip whitespace
        identifiers = [id.strip() for id in text.split(',') if id.strip()]
        return identifiers


class DataVisualizationWidget(QWidget):
    """Widget for visualizing dataset items"""
    
    def __init__(self, raw_data_path):
        super().__init__()
        self.dataset = None
        self.current_index = 0
        self.auto_play = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_item)
        self.current_model = None
        self.expected_sequence_length = None  # Expected sequence length for the loaded model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for inference
        self.raw_data_path = raw_data_path
        self.filtered_track_identifiers = None  # List of track identifiers to filter by
        self.show_filtered_only = False  # Whether to show only filtered tracks or all tracks
        self.filtered_indices = None  # List of indices that match the filter
        self.original_to_filtered_index = {}  # Mapping from original index to filtered index
        # Metrics storage
        self.all_probabilities = None  # Store probabilities from full pass (array)
        self.all_labels = None  # Store labels from full pass (array)
        self.valid_indices = None  # Map from array index to dataset index
        self.predictions_at_threshold = None  # Predictions at best F1 threshold (dict: dataset_idx -> prediction)
        self.best_f1_threshold = None  # Best F1 threshold
        self.best_f1_score = None  # Best F1 score
        self.all_metadata_dict = {}  # Store metadata from full pass (for export without re-running)
        self.all_box_sizes_dict = {}  # Store box sizes for correlation plot
        self.auc_score = None  # AUC score
        self.ap_score = None  # AP score
        self.prediction_filter_mode = "all"  # Filter mode: "all", "only success", "only errors"
        # Model information storage
        self.model_checkpoint_path = None  # Path to loaded checkpoint
        self.model_config = None  # Config dict from checkpoint
        self.model_type = None  # Model type (mlp, lstm, stg_nf)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.prev_item)
        self.prev_btn.setEnabled(False)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_item)
        self.next_btn.setEnabled(False)
        
        self.auto_play_btn = QPushButton("Auto Play")
        self.auto_play_btn.clicked.connect(self.toggle_auto_play)
        self.auto_play_btn.setEnabled(False)

        self.index_label = QLabel("Index: 0/0")
        
        # Dataset index dropdown
        self.index_dropdown = QComboBox()
        self.index_dropdown.setMinimumWidth(300)
        self.index_dropdown.currentIndexChanged.connect(self.on_dropdown_selection_changed)
        
        # Destandardize checkbox (for visualization only)
        self.destandardize_checkbox = QCheckBox("Destandardize")
        self.destandardize_checkbox.setChecked(True)
        self.destandardize_checkbox.toggled.connect(self.on_destandardize_toggled)
        
        # Track filter button
        self.filter_tracks_btn = QPushButton("Filter Tracks")
        self.filter_tracks_btn.clicked.connect(self.open_track_filter_dialog)
        
        # Run a pass button
        self.run_pass_btn = QPushButton("Run a pass")
        self.run_pass_btn.clicked.connect(self.run_full_pass)
        self.run_pass_btn.setEnabled(False)
        
        # Show filtered only checkbox
        self.show_filtered_only_checkbox = QCheckBox("Show Filtered Only")
        self.show_filtered_only_checkbox.setChecked(False)
        self.show_filtered_only_checkbox.setEnabled(False)
        self.show_filtered_only_checkbox.toggled.connect(self.on_filtered_only_toggled)
        
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.next_btn)
        control_layout.addWidget(self.auto_play_btn)
        control_layout.addWidget(QLabel("Jump to:"))
        control_layout.addWidget(self.index_dropdown)
        self.export_track_visu_btn = QPushButton("Export Track Visu")
        self.export_track_visu_btn.clicked.connect(self.export_track_visualization)
        self.export_track_visu_btn.setEnabled(False)
        control_layout.addWidget(self.export_track_visu_btn)
        control_layout.addWidget(self.filter_tracks_btn)
        control_layout.addWidget(self.run_pass_btn)
        control_layout.addWidget(self.show_filtered_only_checkbox)
        control_layout.addStretch()
        control_layout.addWidget(self.destandardize_checkbox)
        control_layout.addWidget(self.index_label)
        
        # Metrics display
        metrics_layout = QHBoxLayout()
        self.metrics_label = QLabel("Metrics: Not computed")
        self.metrics_label.setStyleSheet("font-weight: bold; color: #333; padding: 5px;")
        metrics_layout.addWidget(self.metrics_label)
        
        # Prediction filter dropdown
        self.prediction_filter_dropdown = QComboBox()
        self.prediction_filter_dropdown.addItems(["all", "only success", "only errors"])
        self.prediction_filter_dropdown.setCurrentText("all")
        self.prediction_filter_dropdown.currentTextChanged.connect(self.on_prediction_filter_changed)
        self.prediction_filter_dropdown.setEnabled(False)  # Disabled until predictions are available
        metrics_layout.addWidget(QLabel("Filter:"))
        metrics_layout.addWidget(self.prediction_filter_dropdown)
        
        # Export results button
        self.export_results_btn = QPushButton("Export results")
        self.export_results_btn.clicked.connect(self.export_results)
        self.export_results_btn.setEnabled(False)  # Disabled until predictions are available
        metrics_layout.addWidget(self.export_results_btn)
        metrics_layout.addStretch()
        
        layout.addLayout(control_layout)
        layout.addLayout(metrics_layout)
        
        # Text display (original functionality)
        self.data_display = QTextEdit()
        self.data_display.setReadOnly(True)
        self.data_display.setFont(QFont("Courier", 10))
        self.data_display.setMinimumHeight(400)  # Limit height to make room for graph
        self.data_display.setMaximumHeight(600)  # Limit height to make room for graph
        
        # Graph visualization widget
        self.graph_widget = GraphWidget(self.raw_data_path)
        
        layout.addWidget(self.data_display)
        layout.addWidget(self.graph_widget)
        
        self.setLayout(layout)
        
        # Add keyboard shortcuts
        self.setup_shortcuts()
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts for navigation"""
        # Ctrl+N for next item
        next_shortcut = QShortcut(QKeySequence("Ctrl+N"), self)
        next_shortcut.activated.connect(self.next_item)
        
        # Ctrl+B for previous item (B for "Back")
        prev_shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        prev_shortcut.activated.connect(self.prev_item)
    
    def set_model(self, model, expected_sequence_length=None):
        """Set the model for inference"""
        self.current_model = model
        self.expected_sequence_length = expected_sequence_length
        # Enable run_pass_btn if both model and dataset are available
        self.run_pass_btn.setEnabled(self.current_model is not None and self.dataset is not None and len(self.dataset) > 0)
        if self.dataset is not None and len(self.dataset) > 0:
            # Refresh display to show predictions
            self.display_current_item()
    
    def set_dataset(self, dataset):
        """Set the dataset to visualize"""
        self.dataset = dataset
        self.current_index = 0
        # Reset filter state when new dataset is loaded
        if dataset is not None:
            # Reapply filter if it exists
            if self.filtered_track_identifiers is not None:
                self.apply_track_filter()
            else:
                self.filtered_indices = None
                self.original_to_filtered_index = {}
        else:
            self.filtered_indices = None
            self.original_to_filtered_index = {}
            self.show_filtered_only_checkbox.setEnabled(False)
            self.show_filtered_only_checkbox.setChecked(False)
        
        # Reset metrics when dataset changes
        self.all_probabilities = None
        self.all_labels = None
        self.valid_indices = None
        self.predictions_at_threshold = None
        self.best_f1_threshold = None
        self.best_f1_score = None
        self.all_metadata_dict = {}
        self.all_box_sizes_dict = {}
        self.auc_score = None
        self.ap_score = None
        self.prediction_filter_mode = "all"
        self.prediction_filter_dropdown.setCurrentText("all")
        self.prediction_filter_dropdown.setEnabled(False)
        self.export_results_btn.setEnabled(False)
        self.update_metrics_display()
        
        # Enable run_pass_btn if both model and dataset are available
        self.run_pass_btn.setEnabled(self.current_model is not None and self.dataset is not None and len(self.dataset) > 0)
        # Enable export_track_visu_btn if dataset is available
        self.export_track_visu_btn.setEnabled(self.dataset is not None and len(self.dataset) > 0)
        
        self.update_controls()
        if dataset is not None:
            # Enable/disable destandardize checkbox based on standardization setting
            standardize_data = getattr(dataset, 'standardize_data', 'none')
            self.destandardize_checkbox.setEnabled(standardize_data == "all")
            if standardize_data != "all":
                self.destandardize_checkbox.setChecked(False)
            
            # self.compute_dataset_statistics() # can be slow
            self.populate_index_dropdown()
            self.display_current_item() # will update the self.current_item_input_tensor, self.current_item_label, self.current_item_metadata
            self.graph_widget.set_dataset(dataset)
            self.graph_widget.update_current_data(
                self.current_item_input_tensor, 
                self.current_item_label, 
                self.current_item_metadata,
                self.current_item_images_tensor,
                self.current_item_masks_tensor
            )
        else:
            # Disable destandardize checkbox when no dataset
            self.destandardize_checkbox.setEnabled(False)
            self.destandardize_checkbox.setChecked(False)
    
    def on_destandardize_toggled(self, checked):
        """Refresh visualization when destandardize checkbox is toggled"""
        if self.dataset is not None and len(self.dataset) > 0:
            # Refresh current item display
            self.display_current_item()
            if hasattr(self, 'current_item_input_tensor'):
                self.graph_widget.update_current_data(
                    self.current_item_input_tensor,
                    self.current_item_label,
                    self.current_item_metadata,
                    self.current_item_images_tensor,
                    self.current_item_masks_tensor
                )
    
    def open_track_filter_dialog(self):
        """Open dialog to enter track identifiers for filtering"""
        dialog = TrackFilterDialog(self)
        if dialog.exec():
            track_identifiers = dialog.get_track_identifiers()
            if track_identifiers:
                self.filtered_track_identifiers = track_identifiers
                self.apply_track_filter()
                self.show_filtered_only_checkbox.setEnabled(True)
                QMessageBox.information(
                    self, 
                    "Filter Applied", 
                    f"Filter set for {len(track_identifiers)} track(s).\n"
                    f"Use 'Show Filtered Only' checkbox to toggle between filtered and all tracks."
                )
            else:
                # Clear filter if empty
                self.filtered_track_identifiers = None
                self.filtered_indices = None
                self.original_to_filtered_index = {}
                self.show_filtered_only_checkbox.setEnabled(False)
                self.show_filtered_only_checkbox.setChecked(False)
                self.populate_index_dropdown()
    
    def apply_track_filter(self):
        """Apply the track filter and update the filtered indices"""
        if self.dataset is None or self.filtered_track_identifiers is None:
            return
        
        # Get the mapping data from the dataset
        idx_to_unique_track_identifier = getattr(self.dataset, 'idx_to_unique_track_identifier', [])
        
        # Find indices that match the filtered track identifiers
        self.filtered_indices = []
        self.original_to_filtered_index = {}
        
        for i in range(len(self.dataset)):
            track_id = idx_to_unique_track_identifier[i] if i < len(idx_to_unique_track_identifier) else None
            if track_id in self.filtered_track_identifiers:
                filtered_idx = len(self.filtered_indices)
                self.filtered_indices.append(i)
                self.original_to_filtered_index[i] = filtered_idx
        
        # Update dropdown
        self.populate_index_dropdown()
    
    def on_filtered_only_toggled(self, checked):
        """Handle toggle of 'Show Filtered Only' checkbox"""
        self.show_filtered_only = checked
        self.populate_index_dropdown()
    
    def on_prediction_filter_changed(self, text):
        """Handle prediction filter dropdown change"""
        self.prediction_filter_mode = text
        self.populate_index_dropdown()
    
    def update_metrics_display(self):
        """Update the metrics display label"""
        if self.auc_score is not None and self.ap_score is not None and self.best_f1_score is not None and self.best_f1_threshold is not None:
            self.metrics_label.setText(
                f"Metrics: AUC={self.auc_score:.4f}, AP={self.ap_score:.4f}, "
                f"Best F1={self.best_f1_score:.4f} @ threshold={self.best_f1_threshold:.4f}"
            )
        else:
            self.metrics_label.setText("Metrics: Not computed")
    
    def export_results(self):
        """Export results to a YAML file"""
        if self.predictions_at_threshold is None or self.all_labels is None or self.all_probabilities is None:
            QMessageBox.warning(self, "No Results", "Please run a full pass first to generate results.")
            return
        
        if self.dataset is None:
            QMessageBox.warning(self, "No Dataset", "No dataset available.")
            return
        
        if self.model_checkpoint_path is None:
            QMessageBox.warning(self, "No Model Info", "Model information not available. Please reload the model.")
            return
        
        try:
            # Calculate confusion matrix
            predictions_array = np.array([self.predictions_at_threshold[idx] for idx in sorted(self.predictions_at_threshold.keys())])
            labels_array = np.array([self.all_labels[np.where(self.valid_indices == idx)[0][0]] for idx in sorted(self.predictions_at_threshold.keys())])
            
            tp = np.sum((predictions_array == 1) & (labels_array == 1))
            fp = np.sum((predictions_array == 1) & (labels_array == 0))
            tn = np.sum((predictions_array == 0) & (labels_array == 0))
            fn = np.sum((predictions_array == 0) & (labels_array == 1))
            
            # Get model name from checkpoint path
            model_name = os.path.basename(self.model_checkpoint_path)
            if model_name.endswith('.pth') or model_name.endswith('.pt'):
                model_name = os.path.splitext(model_name)[0]
            
            # Build report structure
            report = {
                'metrics': {
                    'auc': float(self.auc_score) if self.auc_score is not None else None,
                    'ap': float(self.ap_score) if self.ap_score is not None else None,
                    'best_f1': float(self.best_f1_score) if self.best_f1_score is not None else None,
                    'best_f1_threshold': float(self.best_f1_threshold) if self.best_f1_threshold is not None else None,
                },
                'confusion_matrix': {
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn),
                },
                'model_info': {
                    'name': model_name,
                    'type': self.model_type,
                    'checkpoint_path': self.model_checkpoint_path,
                },
                'config': {},
                'item_results': []
            }
            
            # Add config
            if self.model_config is not None:
                # Convert config to YAML-serializable format
                for key, value in self.model_config.items():
                    if isinstance(value, (int, float, str, bool, type(None))):
                        report['config'][key] = value
                    elif isinstance(value, (list, tuple)):
                        report['config'][key] = [v if isinstance(v, (int, float, str, bool)) else str(v) for v in value]
                    elif isinstance(value, dict):
                        report['config'][key] = {k: v if isinstance(v, (int, float, str, bool)) else str(v) for k, v in value.items()}
                    else:
                        report['config'][key] = str(value)
            
            # Get track information for each valid index
            idx_to_unique_track_identifier = getattr(self.dataset, 'idx_to_unique_track_identifier', [])
            
            for dataset_idx in sorted(self.predictions_at_threshold.keys()):
                if dataset_idx >= len(self.dataset):
                    continue
                
                # Use stored metadata instead of calling dataset[idx] again
                if dataset_idx not in self.all_metadata_dict:
                    continue
                metadata = self.all_metadata_dict[dataset_idx]
                
                # Get label from stored results
                idx_in_array = np.where(self.valid_indices == dataset_idx)[0]
                if len(idx_in_array) > 0:
                    label = self.all_labels[idx_in_array[0]]
                    predicted_score = float(self.all_probabilities[idx_in_array[0]])
                else:
                    continue
                
                # Get prediction
                predicted_label = self.predictions_at_threshold[dataset_idx]
                
                # Extract track information from metadata
                unique_track_id = metadata["unique_track_identifier"]
                image_indexes = metadata["image_indexes"]
                xmin_meta = metadata["xmin_meta"]
                xmax_meta = metadata["xmax_meta"]
                ymin_meta = metadata["ymin_meta"]
                ymax_meta = metadata["ymax_meta"]
                time_to_first_interaction = metadata["time_to_first_interaction"]
                
                # Get recording and episode from dataset
                # Access the track data to get recording and episode
                recording = None
                episode = None
                if hasattr(self.dataset, 'datasets_by_unique_track_identifier'):
                    if unique_track_id in self.dataset.datasets_by_unique_track_identifier:
                        track_df = self.dataset.datasets_by_unique_track_identifier[unique_track_id]
                        if len(track_df) > 0:
                            # Get recording and episode from the track dataframe
                            if 'recording' in track_df.columns:
                                recording = str(track_df['recording'].iloc[0])
                            if 'episode' in track_df.columns:
                                episode_val = track_df['episode'].iloc[0]
                                try:
                                    episode = f"{int(episode_val):04d}"
                                except (ValueError, TypeError):
                                    episode = str(episode_val)
                
                # Convert numpy arrays to lists for YAML
                item_result = {
                    'unique_track_id': str(unique_track_id),
                    'recording': recording,
                    'episode': episode,
                    'frame_indexes': [int(idx) for idx in image_indexes] if image_indexes is not None else [],
                    'xmin_meta': [float(x) for x in xmin_meta] if xmin_meta is not None else [],
                    'xmax_meta': [float(x) for x in xmax_meta] if xmax_meta is not None else [],
                    'ymin_meta': [float(y) for y in ymin_meta] if ymin_meta is not None else [],
                    'ymax_meta': [float(y) for y in ymax_meta] if ymax_meta is not None else [],
                    'time_to_first_interaction': float(time_to_first_interaction) if time_to_first_interaction is not None else None,
                    'predicted_score': predicted_score,
                    'predicted_label': int(predicted_label),
                    'used_score_threshold': float(self.best_f1_threshold) if self.best_f1_threshold is not None else None,
                    'ground_truth_label': int(label),
                }
                
                report['item_results'].append(item_result)
            
            # Create output directory
            output_dir = os.path.join(here, "experiments", "visualize_reports")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            model_parent_dir = os.path.basename(os.path.dirname(self.model_checkpoint_path))
            if "expe_" in model_parent_dir:
                model_parent_dir = model_parent_dir.split("_")[:3]
                model_parent_dir = "_".join(model_parent_dir) # just keep the first 3 parts
            filename = f"report_{model_parent_dir}_{model_name}_{timestamp}.yaml"
            output_path = os.path.join(output_dir, filename)
            
            # Write YAML file
            with open(output_path, 'w') as f:
                yaml.dump(report, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
            # Generate correlation plot between box size and predicted score
            graph_path = None
            if hasattr(self, 'all_box_sizes_dict') and self.all_box_sizes_dict:
                try:
                    graph_path = self._export_box_size_correlation_graph(output_dir, model_parent_dir, model_name, timestamp)
                except Exception as graph_error:
                    prWarning(f"Could not generate correlation graph: {graph_error}")
            
            message = f"Results exported to:\n{output_path}\n\nExported {len(report['item_results'])} track results."
            if graph_path:
                message += f"\n\nCorrelation graph saved to:\n{graph_path}"
            
            QMessageBox.information(
                self,
                "Export Successful",
                message
            )
            
        except Exception as e:
            prError(f"Error exporting results: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
    
    def _export_box_size_correlation_graph(self, output_dir, model_parent_dir, model_name, timestamp):
        """Generate and save a correlation graph between box size and predicted score"""
        import matplotlib.pyplot as plt
        from scipy import stats
        
        # Collect data points
        box_sizes = []
        predicted_scores = []
        labels = []
        
        for dataset_idx in sorted(self.predictions_at_threshold.keys()):
            if dataset_idx not in self.all_box_sizes_dict:
                continue
            box_size = self.all_box_sizes_dict[dataset_idx]
            if box_size is None:
                continue
            
            idx_in_array = np.where(self.valid_indices == dataset_idx)[0]
            if len(idx_in_array) == 0:
                continue
            
            predicted_score = self.all_probabilities[idx_in_array[0]]
            label = self.all_labels[idx_in_array[0]]
            
            box_sizes.append(box_size)
            predicted_scores.append(predicted_score)
            labels.append(label)
        
        if len(box_sizes) < 2:
            return None
        
        box_sizes = np.array(box_sizes)
        predicted_scores = np.array(predicted_scores)
        labels = np.array(labels)
        
        # Compute correlation
        correlation, p_value = stats.pearsonr(box_sizes, predicted_scores)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot with different colors for positive and negative samples
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        ax.scatter(box_sizes[neg_mask], predicted_scores[neg_mask], 
                   c='blue', alpha=0.5, label=f'Negative (n={neg_mask.sum()})', s=20)
        ax.scatter(box_sizes[pos_mask], predicted_scores[pos_mask], 
                   c='red', alpha=0.5, label=f'Positive (n={pos_mask.sum()})', s=20)
        
        # Add regression line
        z = np.polyfit(box_sizes, predicted_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(box_sizes.min(), box_sizes.max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, 
                label=f'Trend (r={correlation:.3f}, p={p_value:.2e})')
        
        # Add threshold line if available
        if self.best_f1_threshold is not None:
            ax.axhline(y=self.best_f1_threshold, color='green', linestyle=':', 
                       linewidth=2, label=f'Threshold ({self.best_f1_threshold:.3f})')
        
        ax.set_xlabel('Box Size (pixels²)', fontsize=12)
        ax.set_ylabel('Predicted Score', fontsize=12)
        
        # Build title with time to interaction info
        title_lines = [f'Box Size vs Predicted Score - {model_name}']
        if self.model_config is not None:
            interaction_cutoff = self.model_config["interaction_cutoff_val"]
            positive_cutoff = self.model_config["positive_cutoff_val"]
            if interaction_cutoff is not None or positive_cutoff is not None:
                title_lines.append(f'T_ADV={interaction_cutoff}, T_POS={positive_cutoff}')
        
        ax.set_title('\n'.join(title_lines), fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        ymax = max(1.05, max(predicted_scores))
        ax.set_ylim(-0.05, ymax)
        
        # Save figure
        graph_filename = f"correlation_{model_parent_dir}_{model_name}_{timestamp}.png"
        graph_path = os.path.join(output_dir, graph_filename)
        fig.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return graph_path
    
    def run_full_pass(self):
        """Run a full pass of the model on all dataset samples"""
        if self.current_model is None:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return
        
        if self.dataset is None or len(self.dataset) == 0:
            QMessageBox.warning(self, "No Dataset", "Please create a dataset first.")
            return
        
        # Disable button during computation
        self.run_pass_btn.setEnabled(False)
        self.run_pass_btn.setText("Running...")
        
        # Collect probabilities and labels - store by index
        all_probabilities_dict = {}  # dataset_idx -> probability
        all_labels_dict = {}  # dataset_idx -> label
        all_metadata_dict = {}  # dataset_idx -> metadata (stored for export)
        all_box_sizes_dict = {}  # dataset_idx -> box size (for correlation plot)
        
        try:
            # Run inference on all samples
            for idx in range(len(self.dataset)):
                # Get sample
                sample = self.dataset[idx]
                input_tensor, label, metadata, images_tensor, masks_tensor = sample
                
                input_sequence_length = input_tensor.shape[0]
                
                with torch.no_grad():
                    # Check if we need sliding window inference
                    if self.expected_sequence_length is not None and input_sequence_length != self.expected_sequence_length:
                        if input_sequence_length < self.expected_sequence_length:
                            # Skip samples that are too short - don't store them
                            continue
                        else:
                            # Use sliding window - take final window prediction
                            input_tensor_slide = input_tensor[-self.expected_sequence_length:, :]
                            input_batch = input_tensor_slide.unsqueeze(0).contiguous().to(self.device)
                            
                            if isinstance(self.current_model, STG_NF):
                                input_batch = input_tensor_to_format_by_channel(input_batch, metadata, self.dataset.data_columns_in_dataset)
                                input_batch = keypoints17_to_coco18(input_batch.cpu().numpy())
                                input_batch = np.transpose(input_batch, (0, 3, 1, 2)).astype(np.float32)
                                input_batch = torch.tensor(input_batch).to(self.device)
                                B, C, T, V = input_batch.shape
                                score = torch.ones((B, T), device=self.device).amin(dim=-1)
                                label_model = torch.ones_like(label).to(self.device)
                                z, nll = self.current_model(input_batch[:, :2].float(), label=label_model, score=score, return_z=True)
                                probability = nll.item()

                            elif isinstance(self.current_model, STGCN):
                                # STGCN preprocessing (similar to infer.py)
                                in_channels = self.current_model.in_channels
                                input_batch = input_tensor_to_format_by_channel(input_batch, metadata, self.dataset.data_columns_in_dataset)  # B,T,V,C
                                input_batch = keypoints17_to_coco18_torch(input_batch)  # B,T,V,C with V=18
                                input_batch = input_batch.permute(0, 3, 1, 2).contiguous()  # B,C,T,V with C=18
                                input_batch = input_batch.unsqueeze(-1)  # B,C,T,V,M with M=1
                                input_batch = input_batch[:, :in_channels, :, :, :]  # B,C,T,V,M with C=in_channels
                                input_batch = input_batch.to(self.device)
                                
                                logits = self.current_model(input_batch)
                                probability = torch.sigmoid(logits.squeeze()).item()
                            elif isinstance(self.current_model, SkateFormer):
                                # SkateFormer preprocessing (similar to infer.py)
                                B, T_batch, D = input_batch.shape
                                
                                # Determine input format based on D
                                if D == 56:
                                    input_batch = input_tensor_to_format_by_channel(input_batch, metadata, self.dataset.data_columns_in_dataset)  # B,T,V,C
                                    input_batch = coco2nwucla(input_batch)  # B,T,20,3
                                elif D == 245 or D == 194:
                                    input_batch = input_tensor_to_format_by_channel_sapiens_without_face(input_batch, metadata, self.dataset.data_columns_in_dataset)  # B,T,V,C
                                    input_batch = sapiensnoface2nturgbd_nospine_mid(input_batch)  # B,T,24,3
                                else:
                                    raise ValueError(f"Invalid number of input columns: {D} (expect 56, 245 or 194)")
                                
                                input_batch = input_batch.permute(0, 3, 1, 2).contiguous()  # B,C,T,V
                                input_batch = input_batch.unsqueeze(-1)  # B,C,T,V,M with M=1
                                input_batch = input_batch.to(self.device)
                                
                                # Create index_t tensor
                                T = input_batch.shape[2]
                                index_t = (2 * (torch.arange(0, T, device=self.device) / (T - 1))) - 1  # in [-1, 1]
                                
                                logits = self.current_model(input_batch, index_t=index_t)
                                probability = torch.sigmoid(logits.squeeze()).item()
                            else:
                                logits = self.current_model(input_batch)
                                probability = torch.sigmoid(logits[:, 0]).item()
                    else:
                        # Standard inference
                        input_batch = input_tensor.unsqueeze(0).contiguous().to(self.device)
                        
                        if isinstance(self.current_model, STG_NF):
                            input_batch = input_tensor_to_format_by_channel(input_batch, metadata, self.dataset.data_columns_in_dataset)
                            input_batch = keypoints17_to_coco18(input_batch.cpu().numpy())
                            input_batch = np.transpose(input_batch, (0, 3, 1, 2)).astype(np.float32)
                            input_batch = torch.tensor(input_batch).to(self.device)
                            B, C, T, V = input_batch.shape
                            score = torch.ones((B, T), device=self.device).amin(dim=-1)
                            label_model = torch.ones_like(label).to(self.device)
                            z, nll = self.current_model(input_batch[:, :2].float(), label=label_model, score=score, return_z=True)
                            probability = nll.item()

                        elif isinstance(self.current_model, STGCN):
                            # STGCN preprocessing (similar to infer.py)
                            in_channels = self.current_model.in_channels
                            input_batch = input_tensor_to_format_by_channel(input_batch, metadata, self.dataset.data_columns_in_dataset)  # B,T,V,C
                            input_batch = keypoints17_to_coco18_torch(input_batch)  # B,T,V,C with V=18
                            input_batch = input_batch.permute(0, 3, 1, 2).contiguous()  # B,C,T,V with C=18
                            input_batch = input_batch.unsqueeze(-1)  # B,C,T,V,M with M=1
                            input_batch = input_batch[:, :in_channels, :, :, :]  # B,C,T,V,M with C=in_channels
                            input_batch = input_batch.to(self.device)
                            
                            logits = self.current_model(input_batch)
                            probability = torch.sigmoid(logits.squeeze()).item()
                        elif isinstance(self.current_model, SkateFormer):
                            # SkateFormer preprocessing (similar to infer.py)
                            B, T_batch, D = input_batch.shape
                            
                            # Determine input format based on D
                            if D == 56:
                                input_batch = input_tensor_to_format_by_channel(input_batch, metadata, self.dataset.data_columns_in_dataset)  # B,T,V,C
                                input_batch = coco2nwucla(input_batch)  # B,T,20,3
                            elif D == 245 or D == 194:
                                input_batch = input_tensor_to_format_by_channel_sapiens_without_face(input_batch, metadata, self.dataset.data_columns_in_dataset)  # B,T,V,C
                                input_batch = sapiensnoface2nturgbd_nospine_mid(input_batch)  # B,T,24,3
                            else:
                                raise ValueError(f"Invalid number of input columns: {D} (expect 56, 245 or 194)")
                            
                            input_batch = input_batch.permute(0, 3, 1, 2).contiguous()  # B,C,T,V
                            input_batch = input_batch.unsqueeze(-1)  # B,C,T,V,M with M=1
                            input_batch = input_batch.to(self.device)
                            
                            # Create index_t tensor
                            T = input_batch.shape[2]
                            index_t = (2 * (torch.arange(0, T, device=self.device) / (T - 1))) - 1  # in [-1, 1]
                            
                            logits = self.current_model(input_batch, index_t=index_t)
                            probability = torch.sigmoid(logits.squeeze()).item()
                        else:
                            logits = self.current_model(input_batch)
                            probability = torch.sigmoid(logits.squeeze()).item()
                    
                    all_probabilities_dict[idx] = probability
                    all_labels_dict[idx] = label.item()
                    all_metadata_dict[idx] = metadata  # Store metadata for export
                    
                    # Compute box size from metadata for correlation plot
                    xmin_meta = metadata["xmin_meta"]
                    xmax_meta = metadata["xmax_meta"]
                    ymin_meta = metadata["ymin_meta"]
                    ymax_meta = metadata["ymax_meta"]
                    if len(xmin_meta) > 0 and len(xmax_meta) > 0 and len(ymin_meta) > 0 and len(ymax_meta) > 0:
                        # Use max box size across the sequence
                        widths = np.array(xmax_meta) - np.array(xmin_meta)
                        heights = np.array(ymax_meta) - np.array(ymin_meta)
                        box_sizes = widths * heights
                        all_box_sizes_dict[idx] = float(np.max(box_sizes))
                    else:
                        all_box_sizes_dict[idx] = None
                
                # Process events to keep UI responsive
                QApplication.processEvents()
            
            # Convert to arrays for metric computation (only valid indices)
            all_probabilities_list = []
            all_labels_list = []
            valid_indices = []
            
            for idx in sorted(all_probabilities_dict.keys()):
                if idx in all_labels_dict:
                    all_probabilities_list.append(all_probabilities_dict[idx])
                    all_labels_list.append(all_labels_dict[idx])
                    valid_indices.append(idx)
            
            # Store results as arrays (only for valid indices)
            self.all_probabilities = np.array(all_probabilities_list)
            self.all_labels = np.array(all_labels_list)
            self.valid_indices = np.array(valid_indices)  # Map from array index to dataset index
            self.all_metadata_dict = all_metadata_dict  # Store metadata for export
            self.all_box_sizes_dict = all_box_sizes_dict  # Store box sizes for correlation plot (max box size across the sequence)
            
            # Compute metrics
            if len(self.all_labels) > 0 and len(np.unique(self.all_labels)) > 1:
                self.auc_score = roc_auc_score(self.all_labels, self.all_probabilities)
                self.ap_score = average_precision_score(self.all_labels, self.all_probabilities)
                self.best_f1_threshold, self.best_f1_score = get_best_threshold_f1(
                    self.all_labels, self.all_probabilities, thresholds_min_max=True
                )
                
                # Compute predictions at best threshold
                predictions_array = (self.all_probabilities >= self.best_f1_threshold).astype(int)
                # Map back to dataset indices
                self.predictions_at_threshold = {}
                for i, dataset_idx in enumerate(self.valid_indices):
                    self.predictions_at_threshold[dataset_idx] = predictions_array[i]
            else:
                QMessageBox.warning(self, "Cannot Compute Metrics", "Not enough valid samples or all labels are the same.")
                self.auc_score = None
                self.ap_score = None
                self.best_f1_threshold = None
                self.best_f1_score = None
                self.predictions_at_threshold = None
            
            # Update metrics display
            self.update_metrics_display()
            
            # Enable prediction filter dropdown and export button if predictions are available
            if self.predictions_at_threshold is not None:
                self.prediction_filter_dropdown.setEnabled(True)
                self.export_results_btn.setEnabled(True)
            
            # Update dropdown to show correct/wrong predictions
            self.populate_index_dropdown()
            
            if self.auc_score is not None:
                QMessageBox.information(
                    self,
                    "Full Pass Complete",
                    f"Processed {len(self.all_labels)} samples.\n"
                    f"AUC: {self.auc_score:.4f}\n"
                    f"AP: {self.ap_score:.4f}\n"
                    f"Best F1: {self.best_f1_score:.4f} @ threshold {self.best_f1_threshold:.4f}"
                )
            else:
                QMessageBox.information(
                    self,
                    "Full Pass Complete",
                    f"Processed {len(self.all_labels)} samples.\n"
                    "Could not compute metrics (all labels are the same or insufficient data)."
                )
            
        except Exception as e:
            prError(f"Error during full pass: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to run full pass: {str(e)}")
        finally:
            # Re-enable button
            self.run_pass_btn.setEnabled(True)
            self.run_pass_btn.setText("Run a pass")
    
    def get_display_indices(self):
        """Get the list of indices to display based on filter settings"""
        if self.dataset is None:
            return []
        
        # Start with base indices (filtered tracks or all)
        if self.show_filtered_only and self.filtered_indices is not None:
            base_indices = self.filtered_indices
        else:
            base_indices = list(range(len(self.dataset)))
        
        # Apply prediction filter if predictions are available and filter is not "all"
        if (self.prediction_filter_mode != "all" and 
            self.predictions_at_threshold is not None and 
            isinstance(self.predictions_at_threshold, dict) and
            self.valid_indices is not None and 
            self.all_labels is not None):
            
            filtered_indices = []
            for idx in base_indices:
                if idx in self.predictions_at_threshold:
                    predicted = self.predictions_at_threshold[idx]
                    # Find the corresponding label
                    idx_in_array = np.where(self.valid_indices == idx)[0]
                    if len(idx_in_array) > 0:
                        actual = int(self.all_labels[idx_in_array[0]])
                        is_correct = (predicted == actual)
                        
                        if self.prediction_filter_mode == "only success" and is_correct:
                            filtered_indices.append(idx)
                        elif self.prediction_filter_mode == "only errors" and not is_correct:
                            filtered_indices.append(idx)
                # If index not in predictions, skip it when filtering (only show indices with predictions)
            
            return filtered_indices
        
        return base_indices
    
    def get_original_index(self, display_index):
        """Convert display index to original dataset index"""
        if self.show_filtered_only and self.filtered_indices is not None:
            if display_index < len(self.filtered_indices):
                return self.filtered_indices[display_index]
            else:
                return display_index
        else:
            return display_index
    
    def populate_index_dropdown(self):
        """Populate the index dropdown with dataset indices and their associated track/interaction info"""
        if self.dataset is None:
            self.index_dropdown.clear()
            return
        
        self.index_dropdown.clear()
        
        # Get the mapping data from the dataset
        idx_to_unique_track_identifier = getattr(self.dataset, 'idx_to_unique_track_identifier', [])
        unique_track_to_first_interaction_index = getattr(self.dataset, 'unique_track_to_first_interaction_index', {})
        
        # Get indices to display based on filter
        display_indices = self.get_display_indices()
        
        # Populate dropdown with formatted entries
        for display_idx, original_idx in enumerate(display_indices):
            track_id = idx_to_unique_track_identifier[original_idx] if original_idx < len(idx_to_unique_track_identifier) else "Unknown"
            first_interaction = unique_track_to_first_interaction_index[track_id] if track_id in unique_track_to_first_interaction_index else "Unknown"
            
            # Add correct/wrong status if predictions are available
            prediction_status = ""
            if self.predictions_at_threshold is not None and isinstance(self.predictions_at_threshold, dict):
                if original_idx in self.predictions_at_threshold:
                    predicted = self.predictions_at_threshold[original_idx]
                    # Find the corresponding label
                    if self.valid_indices is not None and self.all_labels is not None:
                        idx_in_array = np.where(self.valid_indices == original_idx)[0]
                        if len(idx_in_array) > 0:
                            actual = int(self.all_labels[idx_in_array[0]])
                            if predicted == actual:
                                prediction_status = " [✓]"
                            else:
                                prediction_status = " [✗]"
            
            # Format the display text
            if self.show_filtered_only:
                display_text = f"Idx {display_idx} (Orig {original_idx}): Track {track_id} (First Interaction: {first_interaction}){prediction_status}"
            else:
                display_text = f"Idx {original_idx}: Track {track_id} (First Interaction: {first_interaction}){prediction_status}"
            self.index_dropdown.addItem(display_text)
        
        # Set current selection (adjust to display index if needed)
        if self.show_filtered_only and self.filtered_indices is not None:
            # Find the display index for current original index
            if self.current_index in self.original_to_filtered_index:
                display_idx = self.original_to_filtered_index[self.current_index]
                self.index_dropdown.setCurrentIndex(display_idx)
            else:
                self.index_dropdown.setCurrentIndex(0)
                self.current_index = self.filtered_indices[0] if self.filtered_indices else 0
        else:
            self.index_dropdown.setCurrentIndex(self.current_index if self.current_index < len(display_indices) else 0)
    
    def on_dropdown_selection_changed(self, index):
        """Handle dropdown selection change"""
        if self.dataset is None:
            return
        
        # Get the original index from the display index
        display_indices = self.get_display_indices()
        if 0 <= index < len(display_indices):
            self.current_index = display_indices[index]
            self.display_current_item() # will update the self.current_item_input_tensor, self.current_item_label, self.current_item_metadata
            self.update_controls()
            self.graph_widget.set_current_index(self.current_index)
            self.graph_widget.update_current_data(
                self.current_item_input_tensor, 
                self.current_item_label, 
                self.current_item_metadata,
                self.current_item_images_tensor,
                self.current_item_masks_tensor
            )
            
    def update_controls(self):
        """Update control button states"""
        has_dataset = self.dataset is not None
        display_indices = self.get_display_indices()
        has_items = has_dataset and len(display_indices) > 0
        
        # Find current display index
        if self.show_filtered_only and self.filtered_indices is not None:
            current_display_idx = self.original_to_filtered_index[self.current_index]
        else:
            current_display_idx = self.current_index
        
        self.prev_btn.setEnabled(has_items and current_display_idx > 0)
        self.next_btn.setEnabled(has_items and current_display_idx < len(display_indices) - 1)
        self.auto_play_btn.setEnabled(has_items)
        self.index_dropdown.setEnabled(has_items)
        
        if has_dataset:
            total_count = len(display_indices)
            display_num = current_display_idx + 1
            if self.show_filtered_only:
                self.index_label.setText(f"Index: {display_num}/{total_count} (Filtered, Orig: {self.current_index + 1})")
            else:
                self.index_label.setText(f"Index: {self.current_index + 1}/{len(self.dataset)}")
        else:
            self.index_label.setText("Index: 0/0")
    
    def display_current_item(self):
        """Display the current dataset item"""
        if self.dataset is None or len(self.dataset) == 0:
            self.data_display.setText("No dataset loaded")
            return
        
        # only call to dataset
        self.current_item_input_tensor, self.current_item_label, self.current_item_metadata, self.current_item_images_tensor, self.current_item_masks_tensor = self.dataset[self.current_index]
        input_tensor = self.current_item_input_tensor
        label = self.current_item_label
        metadata = self.current_item_metadata
        
        
        # metadata_columns = self.dataset.metadata_columns_in_dataset
        data_columns = self.dataset.data_columns_in_dataset
        
        assert(len(data_columns) == input_tensor.shape[1])
        
        # Format the display
        display_text = f"=== Dataset Item {self.current_index + 1} ===\n\n"
        
        # Clear proba_sequence if no model
        if self.current_model is None:
            self.graph_widget.proba_sequence = None
        
        # Perform inference if model is loaded
        model_output_text = ""
        if self.current_model is not None:
            input_sequence_length = input_tensor.shape[0]  # Time dimension
            
            with torch.no_grad():
                # Check if we need sliding window inference
                if self.expected_sequence_length is not None and input_sequence_length != self.expected_sequence_length:
                    # Use sliding window approach (similar to infer.py evaluate_with_display)
                    if input_sequence_length < self.expected_sequence_length:
                        raise ValueError(f"Input sequence length is shorter than expected: {input_sequence_length} < {self.expected_sequence_length}")
                    else:
                        # Input is longer than expected - use sliding window
                        model_output_text += f"Model Prediction (Sliding Window):\n"
                        model_output_text += f"  Input length: {input_sequence_length}, Expected: {self.expected_sequence_length}\n"
                        
                        # Initialize probability sequence (similar to infer.py)
                        # First expected_length frames are WAIT (-1), then probabilities as windows slide
                        proba_sequence = [-1] * self.expected_sequence_length
                        probabilities_list = []
                        
                        # Slide window along the input
                        num_windows = input_sequence_length - self.expected_sequence_length + 1
                        for i in range(num_windows):
                            input_tensor_slide = input_tensor[i:i+self.expected_sequence_length, :]
                            input_batch = input_tensor_slide.unsqueeze(0).contiguous().to(self.device)
                            if isinstance(self.current_model, STG_NF):
                                input_batch = input_tensor_to_format_by_channel(input_batch, metadata, self.dataset.data_columns_in_dataset)
                                input_batch = keypoints17_to_coco18(input_batch.cpu().numpy())
                                input_batch = np.transpose(input_batch, (0, 3, 1, 2)).astype(np.float32)
                                input_batch = torch.tensor(input_batch).to(self.device)
                                B,C,T,V = input_batch.shape # with B=batch, C=channels=3, T=time, V=keypoints=18
                                score = torch.ones((B,T), device=self.device).amin(dim=-1) # to compare against class "normal" distribution
                                label_model = torch.ones_like(label).to(self.device) # at eval time only use ones_like, label_model = torch.where(label_model == 0, 1, -1)
                                z, nll = self.current_model(input_batch[:, :2].float(), label=label_model, score=score, return_z=True)
                                probabilities = nll

                            elif isinstance(self.current_model, STGCN):
                                # STGCN preprocessing (similar to infer.py)
                                in_channels = self.current_model.in_channels
                                input_batch = input_tensor_to_format_by_channel(input_batch, metadata, self.dataset.data_columns_in_dataset)  # B,T,V,C
                                input_batch = keypoints17_to_coco18_torch(input_batch)  # B,T,V,C with V=18
                                input_batch = input_batch.permute(0, 3, 1, 2).contiguous()  # B,C,T,V with C=18
                                input_batch = input_batch.unsqueeze(-1)  # B,C,T,V,M with M=1
                                input_batch = input_batch[:, :in_channels, :, :, :]  # B,C,T,V,M with C=in_channels
                                input_batch = input_batch.to(self.device)
                                
                                logits = self.current_model(input_batch)
                                probabilities = torch.sigmoid(logits.squeeze())
                            elif isinstance(self.current_model, SkateFormer):
                                # SkateFormer preprocessing (similar to infer.py)
                                B, T_batch, D = input_batch.shape
                                
                                # Determine input format based on D
                                if D == 56:
                                    input_batch = input_tensor_to_format_by_channel(input_batch, metadata, self.dataset.data_columns_in_dataset)  # B,T,V,C
                                    input_batch = coco2nwucla(input_batch)  # B,T,20,3
                                elif D == 245 or D == 194:
                                    input_batch = input_tensor_to_format_by_channel_sapiens_without_face(input_batch, metadata, self.dataset.data_columns_in_dataset)  # B,T,V,C
                                    input_batch = sapiensnoface2nturgbd_nospine_mid(input_batch)  # B,T,24,3
                                else:
                                    raise ValueError(f"Invalid number of input columns: {D} (expect 56, 245 or 194)")
                                
                                input_batch = input_batch.permute(0, 3, 1, 2).contiguous()  # B,C,T,V
                                input_batch = input_batch.unsqueeze(-1)  # B,C,T,V,M with M=1
                                input_batch = input_batch.to(self.device)
                                
                                # Create index_t tensor
                                T = input_batch.shape[2]
                                index_t = (2 * (torch.arange(0, T, device=self.device) / (T - 1))) - 1  # in [-1, 1]
                                
                                logits = self.current_model(input_batch, index_t=index_t)
                                probabilities = torch.sigmoid(logits.squeeze())
                            else:
                                logits = self.current_model(input_batch)                                
                                probabilities = torch.sigmoid(logits[:, 0])
                                
                            probability = probabilities.item()
                            proba_sequence.append(probability)
                            probabilities_list.append(probability)
                        
                        # Store proba_sequence for timestamp slider display
                        self.graph_widget.proba_sequence = proba_sequence
                        
                        # Compute statistics
                        avg_probability = np.mean(probabilities_list)
                        max_probability = np.max(probabilities_list)
                        min_probability = np.min(probabilities_list)
                        final_probability = probabilities_list[-1]  # Last window prediction
                        predicted_label = 1 if final_probability >= 0.5 else 0
                        
                        model_output_text += f"  Number of windows: {num_windows}\n"
                        model_output_text += f"  Average Probability: {avg_probability:.4f}\n"
                        model_output_text += f"  Min Probability: {min_probability:.4f}\n"
                        model_output_text += f"  Max Probability: {max_probability:.4f}\n"
                        model_output_text += f"  Final Window Probability: {final_probability:.4f}\n"
                        model_output_text += f"  Predicted Label (final): {predicted_label}\n"
                        model_output_text += f"  Probability Sequence: {[f'{p:.3f}' if p > -1 else 'WAIT' for p in proba_sequence[:10]]}"
                        if len(proba_sequence) > 10:
                            model_output_text += f" ... ({len(proba_sequence)} total)\n"
                        else:
                            model_output_text += "\n"
                else:
                    # Standard inference - sequence lengths match
                    input_batch = input_tensor.unsqueeze(0).contiguous().to(self.device)
                    
                    if isinstance(self.current_model, STG_NF):
                        input_batch = input_tensor_to_format_by_channel(input_batch, metadata, self.dataset.data_columns_in_dataset)
                        input_batch = keypoints17_to_coco18(input_batch.cpu().numpy())
                        input_batch = np.transpose(input_batch, (0, 3, 1, 2)).astype(np.float32)
                        input_batch = torch.tensor(input_batch).to(self.device)
                        B,C,T,V = input_batch.shape # with B=batch, C=channels=3, T=time, V=keypoints=18
                        score = torch.ones((B,T), device=self.device).amin(dim=-1) # to compare against class "normal" distribution
                        label_model = torch.ones_like(label).to(self.device) # at eval time only use ones_like, label_model = torch.where(label_model == 0, 1, -1)
                        z, nll = self.current_model(input_batch[:, :2].float(), label=label_model, score=score, return_z=True)
                        model_output = nll.item()
                        probabilities = nll
                        probability = probabilities.item()

                    elif isinstance(self.current_model, STGCN):
                        # STGCN preprocessing (similar to infer.py)
                        in_channels = self.current_model.in_channels
                        input_batch = input_tensor_to_format_by_channel(input_batch, metadata, self.dataset.data_columns_in_dataset)  # B,T,V,C
                        input_batch = keypoints17_to_coco18_torch(input_batch)  # B,T,V,C with V=18
                        input_batch = input_batch.permute(0, 3, 1, 2).contiguous()  # B,C,T,V with C=18
                        input_batch = input_batch.unsqueeze(-1)  # B,C,T,V,M with M=1
                        input_batch = input_batch[:, :in_channels, :, :, :]  # B,C,T,V,M with C=in_channels
                        input_batch = input_batch.to(self.device)
                        
                        logits = self.current_model(input_batch)
                        model_output = logits.squeeze().item()
                        probability = torch.sigmoid(torch.tensor(model_output)).item()
                    elif isinstance(self.current_model, SkateFormer):
                        # SkateFormer preprocessing (similar to infer.py)
                        B, T_batch, D = input_batch.shape
                        
                        # Determine input format based on D
                        if D == 56:
                            input_batch = input_tensor_to_format_by_channel(input_batch, metadata, self.dataset.data_columns_in_dataset)  # B,T,V,C
                            input_batch = coco2nwucla(input_batch)  # B,T,20,3
                        elif D == 245 or D == 194:
                            input_batch = input_tensor_to_format_by_channel_sapiens_without_face(input_batch, metadata, self.dataset.data_columns_in_dataset)  # B,T,V,C
                            input_batch = sapiensnoface2nturgbd_nospine_mid(input_batch)  # B,T,24,3
                        else:
                            raise ValueError(f"Invalid number of input columns: {D} (expect 56, 245 or 194)")
                        
                        input_batch = input_batch.permute(0, 3, 1, 2).contiguous()  # B,C,T,V
                        input_batch = input_batch.unsqueeze(-1)  # B,C,T,V,M with M=1
                        input_batch = input_batch.to(self.device)
                        
                        # Create index_t tensor
                        T = input_batch.shape[2]
                        index_t = (2 * (torch.arange(0, T, device=self.device) / (T - 1))) - 1  # in [-1, 1]
                        
                        logits = self.current_model(input_batch, index_t=index_t)
                        model_output = logits.squeeze().item()
                        probability = torch.sigmoid(torch.tensor(model_output)).item()
                    else:
                        model_output = self.current_model(input_batch)
                        model_output = model_output.squeeze().item()
                        # Apply sigmoid to get probability
                        probability = torch.sigmoid(torch.tensor(model_output)).item()
                        
                    predicted_label = 1 if probability >= 0.5 else 0
                        
                    
                    # Store proba_sequence for timestamp slider display (same probability for all frames)
                    if self.expected_sequence_length is not None:
                        self.graph_widget.proba_sequence = [probability] * self.expected_sequence_length
                    else:
                        self.graph_widget.proba_sequence = [probability] * input_sequence_length
                    
                    model_output_text += f"Model Prediction:\n"
                    model_output_text += f"  Logit: {model_output:.4f}\n"
                    model_output_text += f"  Probability: {probability:.4f}\n"
                    model_output_text += f"  Predicted Label: {predicted_label}\n"
                
                model_output_text += f"  Ground Truth Label: {label.item()}\n"
                # For sliding window, compare final prediction with ground truth
                if self.expected_sequence_length is not None and input_sequence_length > self.expected_sequence_length:
                    match = '✓' if predicted_label == label.item() else '✗'
                    model_output_text += f"  Match (final): {match}\n\n"
                else:
                    match = '✓' if predicted_label == label.item() else '✗'
                    model_output_text += f"  Match: {match}\n\n"

        
        display_text += model_output_text
        display_text += f"Label: {label.item()}\n"
        display_text += f"Input Shape: {input_tensor.shape}\n\n"
        

        display_text += "Metadata:\n"
        for key, value in metadata.items():
            display_text += f"  {key}: {value}\n"
        
        display_text += f"\nInput Data (first 10 values):\n"
        input_data = input_tensor.numpy()
        if input_data.size > 0:
            display_text += f"  {input_data.flatten()[:10]}\n"
        
        display_text += f"\nInput Statistics:\n"
        display_text += f"  Min: {input_data.min():.4f}\n"
        display_text += f"  Max: {input_data.max():.4f}\n"
        display_text += f"  Mean: {input_data.mean():.4f}\n"
        display_text += f"  Std: {input_data.std():.4f}\n"

        # display_text += f"Metadata Columns: {metadata_columns}\n"
        display_text += f"Data Columns: {data_columns}\n"
        
        self.data_display.setText(display_text)
    
    def prev_item(self):
        """Go to previous item"""
        display_indices = self.get_display_indices()
        if len(display_indices) == 0:
            return
        
        # Find current display index
        if self.show_filtered_only and self.filtered_indices is not None:
            current_display_idx = self.original_to_filtered_index[self.current_index]
        else:
            current_display_idx = display_indices.index(self.current_index) if self.current_index in display_indices else 0
        
        if current_display_idx > 0:
            # Move to previous display index
            prev_display_idx = current_display_idx - 1
            self.current_index = display_indices[prev_display_idx]
            self.graph_widget.set_current_index(self.current_index)
            self.index_dropdown.setCurrentIndex(prev_display_idx) # will call display_current_item and update_controls via on_dropdown_selection_changed
    
    def next_item(self):
        """Go to next item"""
        display_indices = self.get_display_indices()
        if len(display_indices) == 0:
            return
        
        # Find current display index
        if self.show_filtered_only and self.filtered_indices is not None:
            current_display_idx = self.original_to_filtered_index[self.current_index]
        else:
            current_display_idx = display_indices.index(self.current_index) if self.current_index in display_indices else 0
        
        if current_display_idx < len(display_indices) - 1:
            # Move to next display index
            next_display_idx = current_display_idx + 1
            self.current_index = display_indices[next_display_idx]
            self.graph_widget.set_current_index(self.current_index)
            self.index_dropdown.setCurrentIndex(next_display_idx) # will call display_current_item and update_controls via on_dropdown_selection_changed
    
    def toggle_auto_play(self):
        """Toggle auto-play mode"""
        self.auto_play = not self.auto_play
        if self.auto_play:
            self.auto_play_btn.setText("Stop Auto Play")
            self.timer.start(100)  # 100ms interval
        else:
            self.auto_play_btn.setText("Auto Play")
            self.timer.stop()
    
    def export_track_visualization(self):
        """Export a video visualization of the current track"""
        if self.dataset is None or self.current_index >= len(self.dataset):
            QMessageBox.warning(self, "Export Error", "No dataset or track selected.")
            return
        
        # Get current track data
        current_data = self.graph_widget.current_data
        if current_data is None or len(current_data) == 0:
            QMessageBox.warning(self, "Export Error", "No data available for current track.")
            return
        
        # Get metadata
        metadata = self.current_item_metadata
        if metadata is None:
            QMessageBox.warning(self, "Export Error", "No metadata available for current track.")
            return
        
        # Get unique track identifier
        unique_track_identifier = metadata["unique_track_identifier"]
        if isinstance(unique_track_identifier, torch.Tensor):
            if unique_track_identifier.dim() > 0 and len(unique_track_identifier) > 0:
                unique_track_identifier = str(unique_track_identifier[0].item())
            else:
                unique_track_identifier = str(unique_track_identifier.item())
        else:
            unique_track_identifier = str(unique_track_identifier)
        
        # Get experiment name and checkpoint info
        experiment_name = "no_expe"
        checkpoint_filename = "N/A"
        if self.model_checkpoint_path:
            checkpoint_filename = os.path.basename(self.model_checkpoint_path)
            model_parent_dir = os.path.basename(os.path.dirname(self.model_checkpoint_path))
            if "expe_" in model_parent_dir:
                experiment_name = "_".join(model_parent_dir.split("_")[:3])
            else:
                experiment_name = model_parent_dir
        
        # Create output directory
        output_dir = os.path.join(here, "experiments", "visualize_tracks")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"track_visu_{unique_track_identifier}_{timestamp}_{experiment_name}.mp4"
        file_path = os.path.join(output_dir, filename)
        
        # Get normalization parameters
        normalize_in_image = getattr(self.dataset, 'normalize_in_image', False)
        normalize_keypoints_in_box = getattr(self.dataset, 'normalize_keypoints_in_box', False)
        normalize_keypoints_in_track = getattr(self.dataset, 'normalize_keypoints_in_track', 'none')
        
        # Get center_on_onset and interaction_cutoff
        center_on_onset = getattr(self.dataset, 'center_on_onset', False)
        interaction_cutoff = getattr(self.dataset, 'interaction_cutoff', None)
        
        # Get cross_eval_type from config
        cross_eval_type = "N/A"
        if self.model_config is not None:
            cross_eval_type = self.model_config["cross_eval_type"]
            if cross_eval_type is None:
                cross_eval_type = "N/A"
        
        # Get time to first interaction from metadata
        # This is the time from the END of the sequence to the first interaction
        time_to_first_interaction = metadata["time_to_first_interaction"]
        if isinstance(time_to_first_interaction, torch.Tensor):
            time_to_first_interaction = time_to_first_interaction.item()
        
        # Get destandardize setting
        destandardize = self.destandardize_checkbox.isChecked()
        
        # Get images and masks
        images_tensor = self.current_item_images_tensor
        masks_tensor = self.current_item_masks_tensor
        
        # Get proba_sequence if available
        proba_sequence = self.graph_widget.proba_sequence
        
        # Get image dimensions
        image_width, image_height = self.graph_widget.get_image_dimensions()
        is_normalized = self.graph_widget.is_data_normalized()
        
        # Determine video dimensions
        # Header height: 120px (for 2 lines with larger text)
        # Left side (image): original image size
        # Right side (skeleton+box): same height as image, width: 1600px (twice as wide)
        header_height = 120
        num_frames = len(current_data)
        
        if images_tensor is not None and len(images_tensor) > 0:
            # Get first image to determine dimensions
            first_image = images_tensor[0]
            if isinstance(first_image, torch.Tensor):
                img_h, img_w = first_image.shape[1], first_image.shape[2]
            else:
                img_h, img_w = first_image.shape[0], first_image.shape[1]
        else:
            img_h, img_w = 1080, 1920  # Default dimensions
        
        right_panel_width = 1600  # Twice as wide (was 800)
        video_width = img_w + right_panel_width
        video_height = header_height + img_h
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10.0  # 10 frames per second
        out = cv2.VideoWriter(file_path, fourcc, fps, (video_width, video_height))
        
        if not out.isOpened():
            QMessageBox.critical(self, "Export Error", f"Failed to create video writer for: {file_path}")
            return
        
        # Progress dialog
        progress = QProgressBar()
        progress.setMaximum(num_frames)
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Exporting Track Visualization")
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(QLabel(f"Exporting {num_frames} frames..."))
        progress_layout.addWidget(progress)
        progress_dialog.setLayout(progress_layout)
        progress_dialog.show()
        QApplication.processEvents()
        
        try:
            for frame_idx in range(num_frames):
                # Create blank frame
                frame = np.ones((video_height, video_width, 3), dtype=np.uint8) * 255
                
                # Draw header with metadata
                # Text is twice as big, and lines are arranged side by side
                header_y = 30
                font_scale = 1.2  # Twice as big (was 0.6)
                thickness = 2
                color = (0, 0, 0)
                
                interaction_cutoff_str = f"{interaction_cutoff} frames" if interaction_cutoff is not None else "N/A"
                
                # Calculate per-frame time to interaction
                # time_to_first_interaction is the TTI at the END of the sequence (last frame)
                # So for frame_idx, the TTI is: time_to_first_interaction + (num_frames - 1 - frame_idx)
                if time_to_first_interaction is not None and not np.isinf(time_to_first_interaction):
                    tti_at_frame = time_to_first_interaction + (num_frames - 1 - frame_idx)
                    tti_str = f"TTI: {tti_at_frame:.0f}"
                elif time_to_first_interaction is not None and np.isinf(time_to_first_interaction):
                    tti_str = "TTI: ∞ (no interaction)"
                else:
                    tti_str = "TTI: N/A"
                
                # Line 1: Experiment/Checkpoint on left, Track ID/Frame/TTI on right
                line1_left = f"Experiment: {experiment_name} | Checkpoint: {checkpoint_filename}"
                line1_right = f"Track: {unique_track_identifier} | Frame: {frame_idx}/{num_frames-1} | {tti_str}"
                
                # Line 2: Normalization on left, Center on Onset/Interaction Cutoff/Cross Eval on right
                line2_left = f"Normalization: in_image={normalize_in_image}, in_box={normalize_keypoints_in_box}, in_track={normalize_keypoints_in_track}"
                line2_right = f"Center on Onset: {center_on_onset} | Interaction Cutoff: {interaction_cutoff_str} | Cross Eval: {cross_eval_type}"
                
                # Calculate x position for right side (approximately middle of video width)
                mid_x = video_width // 2
                
                # Draw line 1 (side by side)
                cv2.putText(frame, line1_left, (10, header_y), cv2.FONT_HERSHEY_SIMPLEX, 
                          font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, line1_right, (mid_x, header_y), cv2.FONT_HERSHEY_SIMPLEX, 
                          font_scale, color, thickness, cv2.LINE_AA)
                
                # Draw line 2 (side by side)
                line2_y = header_y + 50
                cv2.putText(frame, line2_left, (10, line2_y), cv2.FONT_HERSHEY_SIMPLEX, 
                          font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, line2_right, (mid_x, line2_y), cv2.FONT_HERSHEY_SIMPLEX, 
                          font_scale, color, thickness, cv2.LINE_AA)
                
                # Draw image with box overlay (left side)
                image_np = None
                if images_tensor is not None and frame_idx < len(images_tensor):
                    # Use image from tensor
                    image_tensor = images_tensor[frame_idx]
                    if isinstance(image_tensor, torch.Tensor):
                        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
                    else:
                        image_np = np.array(image_tensor)
                        if image_np.shape[0] == 3:
                            image_np = np.transpose(image_np, (1, 2, 0))
                    
                    # Normalize to [0, 255]
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                    else:
                        image_np = image_np.astype(np.uint8)
                elif self.raw_data_path and metadata:
                    # Load image from video file (like plot_image does)
                    try:
                        unique_track_id = metadata["unique_track_identifier"]
                        if isinstance(unique_track_id, torch.Tensor):
                            if unique_track_id.dim() > 0 and len(unique_track_id) > 0:
                                unique_track_id = str(unique_track_id[0].item())
                            else:
                                unique_track_id = str(unique_track_id.item())
                        else:
                            unique_track_id = str(unique_track_id)
                        
                        parts = unique_track_id.rsplit("_", 2)
                        recording = str(parts[0])
                        episode = str(parts[1])
                        
                        image_indexes = metadata["image_indexes"]
                        if image_indexes and frame_idx < len(image_indexes):
                            video_dir = os.path.join(self.raw_data_path, "video_mini", recording, "episodes", episode)
                            video_path = os.path.join(video_dir, "images_360_mini.mp4")
                            
                            if os.path.exists(video_path):
                                cap = cv2.VideoCapture(video_path)
                                cap.set(cv2.CAP_PROP_POS_FRAMES, image_indexes[frame_idx])
                                ret, frame_bgr = cap.read()
                                cap.release()
                                if ret:
                                    image_np = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        prWarning(f"Failed to load image from video: {e}")
                
                if image_np is not None:
                    # Resize if needed
                    if image_np.shape[:2] != (img_h, img_w):
                        image_np = cv2.resize(image_np, (img_w, img_h))
                    
                    # Place image on left side
                    frame[header_height:header_height+img_h, 0:img_w] = image_np
                
                # Check if shift was applied (do_recenter_interaction_zone was used)
                shift_applied = metadata["shift_applied"]
                interaction_zone_center_positions = metadata["interaction_zone_center_position"]
                xmin_meta = metadata["xmin_meta"]
                xmax_meta = metadata["xmax_meta"]
                ymin_meta = metadata["ymin_meta"]
                ymax_meta = metadata["ymax_meta"]
                image_width = metadata["image_size"][0]
                image_height = metadata["image_size"][1]
                
                # Convert tensors to numpy arrays if needed
                if xmin_meta is not None:
                    if isinstance(xmin_meta, torch.Tensor):
                        xmin_meta = xmin_meta.cpu().numpy()
                    elif isinstance(xmin_meta, list):
                        xmin_meta = np.array(xmin_meta)
                if xmax_meta is not None:
                    if isinstance(xmax_meta, torch.Tensor):
                        xmax_meta = xmax_meta.cpu().numpy()
                    elif isinstance(xmax_meta, list):
                        xmax_meta = np.array(xmax_meta)
                if ymin_meta is not None:
                    if isinstance(ymin_meta, torch.Tensor):
                        ymin_meta = ymin_meta.cpu().numpy()
                    elif isinstance(ymin_meta, list):
                        ymin_meta = np.array(ymin_meta)
                if ymax_meta is not None:
                    if isinstance(ymax_meta, torch.Tensor):
                        ymax_meta = ymax_meta.cpu().numpy()
                    elif isinstance(ymax_meta, list):
                        ymax_meta = np.array(ymax_meta)
                if interaction_zone_center_positions is not None:
                    if isinstance(interaction_zone_center_positions, torch.Tensor):
                        interaction_zone_center_positions = interaction_zone_center_positions.cpu().numpy()
                    elif isinstance(interaction_zone_center_positions, list):
                        interaction_zone_center_positions = np.array(interaction_zone_center_positions)
                
                has_shift = (shift_applied is not None and len(shift_applied) > frame_idx and 
                            xmin_meta is not None and len(xmin_meta) > frame_idx)
                
                # Draw original bounding box from metadata if shift was applied
                if has_shift:
                    # Get original bounding box coordinates from metadata
                    orig_xmin = xmin_meta[frame_idx]/image_width
                    orig_xmax = xmax_meta[frame_idx]/image_width
                    orig_ymin = ymin_meta[frame_idx]/image_height
                    orig_ymax = ymax_meta[frame_idx]/image_height
                    
                    if not (np.isnan(orig_xmin) or np.isnan(orig_xmax) or np.isnan(orig_ymin) or np.isnan(orig_ymax)):
                        # Convert to pixel coordinates
                        if is_normalized:
                            orig_xmin_px = int(orig_xmin * img_w)
                            orig_xmax_px = int(orig_xmax * img_w)
                            orig_ymin_px = int(orig_ymin * img_h)
                            orig_ymax_px = int(orig_ymax * img_h)
                        else:
                            orig_xmin_px = int(orig_xmin)
                            orig_xmax_px = int(orig_xmax)
                            orig_ymin_px = int(orig_ymin)
                            orig_ymax_px = int(orig_ymax)
                        
                        # Draw original bounding box (green, solid, normal thickness)
                        cv2.rectangle(frame, 
                                    (orig_xmin_px, header_height + orig_ymin_px),
                                    (orig_xmax_px, header_height + orig_ymax_px),
                                    (0, 255, 0), 2)
                
                # Draw bounding box overlay (shifted if applicable)
                if frame_idx < len(current_data):
                    frame_data = current_data.iloc[frame_idx]
                    if 'xmin' in frame_data and 'xmax' in frame_data and 'ymin' in frame_data and 'ymax' in frame_data:
                        xmin = frame_data['xmin']
                        xmax = frame_data['xmax']
                        ymin = frame_data['ymin']
                        ymax = frame_data['ymax']
                        
                        if not (np.isnan(xmin) or np.isnan(xmax) or np.isnan(ymin) or np.isnan(ymax)):
                            # Convert normalized coordinates to pixel coordinates
                            if is_normalized:
                                xmin_px = int(xmin * img_w)
                                xmax_px = int(xmax * img_w)
                                ymin_px = int(ymin * img_h)
                                ymax_px = int(ymax * img_h)
                            else:
                                xmin_px = int(xmin)
                                xmax_px = int(xmax)
                                ymin_px = int(ymin)
                                ymax_px = int(ymax)
                            
                            # Draw bounding box (shifted: dashed and thinner, or normal: solid)
                            if has_shift:
                                # Draw dashed rectangle for shifted box
                                # OpenCV doesn't support dashed directly, so we draw multiple small lines
                                dash_length = 10
                                gap_length = 5
                                
                                # Top edge
                                x = xmin_px
                                while x < xmax_px:
                                    end_x = min(x + dash_length, xmax_px)
                                    cv2.line(frame, (x, header_height + ymin_px), 
                                           (end_x, header_height + ymin_px), (0, 255, 0), 1)
                                    x += dash_length + gap_length
                                
                                # Bottom edge
                                x = xmin_px
                                while x < xmax_px:
                                    end_x = min(x + dash_length, xmax_px)
                                    cv2.line(frame, (x, header_height + ymax_px), 
                                           (end_x, header_height + ymax_px), (0, 255, 0), 1)
                                    x += dash_length + gap_length
                                
                                # Left edge
                                y = ymin_px
                                while y < ymax_px:
                                    end_y = min(y + dash_length, ymax_px)
                                    cv2.line(frame, (xmin_px, header_height + y), 
                                           (xmin_px, header_height + end_y), (0, 255, 0), 1)
                                    y += dash_length + gap_length
                                
                                # Right edge
                                y = ymin_px
                                while y < ymax_px:
                                    end_y = min(y + dash_length, ymax_px)
                                    cv2.line(frame, (xmax_px, header_height + y), 
                                           (xmax_px, header_height + end_y), (0, 255, 0), 1)
                                    y += dash_length + gap_length
                            else:
                                # Normal solid rectangle
                                cv2.rectangle(frame, 
                                            (xmin_px, header_height + ymin_px),
                                            (xmax_px, header_height + ymax_px),
                                            (0, 255, 0), 2)
                
                # Draw interaction zone center position icon at the bottom
                if interaction_zone_center_positions is not None and len(interaction_zone_center_positions) > frame_idx:
                    iz_center_x = interaction_zone_center_positions[frame_idx] / image_width
                    if isinstance(iz_center_x, torch.Tensor):
                        iz_center_x = iz_center_x.item()
                    if not np.isnan(iz_center_x) and iz_center_x >= 0:
                        # Convert to pixel coordinates
                        # interaction_zone_center_position is in pixel coordinates (not normalized)
                        if iz_center_x <= 1.0:
                            iz_center_x_px = int(iz_center_x * img_w)
                        else:
                            iz_center_x_px = int(iz_center_x)
                        
                        # Draw small icon at bottom of image (small circle)
                        icon_y = header_height + img_h - 20  # 20 pixels from bottom
                        cv2.circle(frame, (iz_center_x_px, icon_y), 8, (0, 0, 255), -1)  # Red filled circle
                        cv2.circle(frame, (iz_center_x_px, icon_y), 8, (139, 0, 0), 2)  # Dark red border
                        # Add small vertical line to show it's at the bottom
                        cv2.line(frame, (iz_center_x_px, icon_y), 
                               (iz_center_x_px, header_height + img_h), 
                               (0, 0, 255), 2)
                
                # Draw virtual center position icon (where interaction zone gets recentered to) at the bottom
                if has_shift:
                    # Virtual center is at the center of the image
                    virtual_center_x_px = img_w // 2
                    icon_y = header_height + img_h - 20  # 20 pixels from bottom
                    # Draw dashed circle (using multiple small arcs to simulate dashed)
                    # Draw circle outline with gaps to create dashed effect
                    radius = 8
                    num_segments = 16
                    for i in range(num_segments):
                        if i % 2 == 0:  # Draw every other segment to create dashed effect
                            angle1 = (i * 2 * np.pi / num_segments)
                            angle2 = ((i + 1) * 2 * np.pi / num_segments)
                            x1 = int(virtual_center_x_px + radius * np.cos(angle1))
                            y1 = int(icon_y + radius * np.sin(angle1))
                            x2 = int(virtual_center_x_px + radius * np.cos(angle2))
                            y2 = int(icon_y + radius * np.sin(angle2))
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # Add small vertical dashed line to show it's at the bottom
                    dash_length = 5
                    gap_length = 3
                    y = icon_y
                    while y < header_height + img_h:
                        end_y = min(y + dash_length, header_height + img_h)
                        cv2.line(frame, (virtual_center_x_px, y), 
                               (virtual_center_x_px, end_y), 
                               (0, 0, 255), 2)
                        y += dash_length + gap_length
                
                # Draw skeleton and box visualization (right side)
                right_x_start = img_w
                right_y_start = header_height
                
                # Check what to show
                show_vitpose = self.graph_widget.vitpose_skeleton_checkbox.isChecked()
                show_sapiens = self.graph_widget.sapiens_skeleton_checkbox.isChecked()
                show_bounding_box = self.graph_widget.show_bounding_box_checkbox.isChecked()
                
                # Create a matplotlib figure for skeleton/box visualization
                # Figure size matches the right panel width (now 1600px)
                fig = plt.Figure(figsize=(right_panel_width/100, img_h/100), dpi=100)
                
                # Determine subplot layout
                if show_bounding_box and (show_vitpose or show_sapiens):
                    # Two subplots side by side
                    ax_skeleton = fig.add_subplot(121)
                    ax_bbox = fig.add_subplot(122)
                elif show_bounding_box:
                    # Only bounding box
                    ax_skeleton = None
                    ax_bbox = fig.add_subplot(111)
                elif show_vitpose or show_sapiens:
                    # Only skeleton
                    ax_skeleton = fig.add_subplot(111)
                    ax_bbox = None
                else:
                    # Nothing to show, create empty subplot
                    ax_skeleton = None
                    ax_bbox = fig.add_subplot(111)
                    ax_bbox.text(0.5, 0.5, 'Enable skeleton or bounding box\nto see visualization', 
                                ha='center', va='center', transform=ax_bbox.transAxes)
                    ax_bbox.axis('off')
                
                # Temporarily set current_data in graph_widget for plotting methods
                original_current_data = self.graph_widget.current_data
                self.graph_widget.current_data = current_data
                
                try:
                    # Plot bounding box
                    if show_bounding_box and ax_bbox is not None and frame_idx < len(current_data):
                        self.graph_widget.plot_bounding_box(ax_bbox, frame_idx, is_normalized, image_width, image_height)
                    
                    # Plot skeleton if enabled
                    if show_vitpose and ax_skeleton is not None and frame_idx < len(current_data):
                        vitpose_connections = [
                            (0, 1), (0, 2), (1, 3), (2, 4),
                            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                            (5, 11), (6, 12), (11, 12),
                            (11, 13), (13, 15), (12, 14), (14, 16)
                        ]
                        keypoints_colors = [(c[0]/255, c[1]/255, c[2]/255) for c in VITPOSE_COLORS]
                        self.graph_widget.plot_skeleton(ax_skeleton, frame_idx, 'vitpose', 
                                                       VITPOSE_KEYPOINTS_NAMES, vitpose_connections,
                                                       'red', keypoints_colors, 'VitPose', 
                                                       is_normalized, image_width, image_height)
                    
                    if show_sapiens and ax_skeleton is not None and frame_idx < len(current_data):
                        sapiens_connections = [GOLIATH_SKELETON_INFO[i]['link'] for i in range(len(GOLIATH_SKELETON_INFO))]
                        keypoints_colors = [(c[0]/255, c[1]/255, c[2]/255) for c in GOLIATH_KPTS_COLORS]
                        self.graph_widget.plot_skeleton(ax_skeleton, frame_idx, 'sapiens_308',
                                                       GOLIATH_KEYPOINTS_NAMES, sapiens_connections,
                                                       'blue', keypoints_colors, 'Sapiens',
                                                       is_normalized, image_width, image_height)
                    
                    # Set up axes
                    axes_to_setup = []
                    if ax_skeleton is not None:
                        axes_to_setup.append(ax_skeleton)
                    if ax_bbox is not None:
                        axes_to_setup.append(ax_bbox)
                    
                    for ax in axes_to_setup:
                        if self.graph_widget.is_standardized() and destandardize:
                            if is_normalized:
                                ax.set_xlim(0, 1)
                                ax.set_ylim(0, 1)
                            else:
                                ax.set_xlim(0, image_width)
                                ax.set_ylim(0, image_height)
                        else:
                            if is_normalized:
                                ax.set_xlim(0, 1)
                                ax.set_ylim(0, 1)
                            else:
                                ax.set_xlim(0, image_width)
                                ax.set_ylim(0, image_height)
                        ax.set_aspect('equal')
                        ax.invert_yaxis()
                        ax.grid(True, alpha=0.3)
                
                finally:
                    # Restore original current_data
                    self.graph_widget.current_data = original_current_data
                
                fig.tight_layout()
                
                # Convert matplotlib figure to numpy array
                # Use a reliable method: save to buffer and read with cv2
                fig_buffer = io.BytesIO()
                # Save with exact dimensions (no tight bbox to preserve size)
                fig.savefig(fig_buffer, format='png', bbox_inches='tight', pad_inches=0, 
                           dpi=100, facecolor='white', edgecolor='none')
                fig_buffer.seek(0)
                # Read image from buffer using cv2
                img_array = np.frombuffer(fig_buffer.getvalue(), dtype=np.uint8)
                fig_buffer.close()
                # Decode PNG to numpy array
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    # Convert BGR to RGB
                    buf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    # Fallback: create blank image
                    buf = np.ones((img_h, right_panel_width, 3), dtype=np.uint8) * 255
                plt.close(fig)
                
                # Resize to fit right panel
                buf_resized = cv2.resize(buf, (right_panel_width, img_h))
                frame[right_y_start:right_y_start+img_h, right_x_start:right_x_start+right_panel_width] = buf_resized
                
                # Add prediction info overlay
                if proba_sequence is not None and frame_idx < len(proba_sequence):
                    proba = proba_sequence[frame_idx]
                    if proba > -1:
                        pred_text = f"Prediction: {proba:.3f} ({proba*100:.1f}%)"
                        if self.predictions_at_threshold is not None and self.current_index in self.predictions_at_threshold:
                            pred_label = self.predictions_at_threshold[self.current_index]
                            pred_text += f" [Label: {pred_label}]"
                    else:
                        pred_text = "Prediction: WAIT"
                    
                    # Draw prediction text on right side (3 times bigger: 0.7 * 3 = 2.1)
                    cv2.putText(frame, pred_text, (right_x_start + 10, header_height + 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 2.1, (255, 0, 0), 4, cv2.LINE_AA)
                
                # Write frame to video
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                
                # Update progress
                progress.setValue(frame_idx + 1)
                QApplication.processEvents()
            
            out.release()
            progress_dialog.close()
            QMessageBox.information(self, "Export Complete", 
                                  f"Track visualization exported successfully to:\n{file_path}")
        
        except Exception as e:
            out.release()
            progress_dialog.close()
            QMessageBox.critical(self, "Export Error", f"Error during export: {str(e)}")
            import traceback
            traceback.print_exc()


class ParameterControlsWidget(QWidget):
    """Widget for dataset parameter controls"""
    
    def __init__(self):
        super().__init__()
        self.model_loaded = False  # Track if model is loaded to freeze controls
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Create scroll area for parameters
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # Include recordings group
        recordings_group = QGroupBox("Include Recordings")
        recordings_layout = QVBoxLayout()
        
        # Selection mode radio buttons
        selection_mode_layout = QHBoxLayout()
        self.selection_mode_group = QButtonGroup()
        self.select_by_recording_radio = QRadioButton("Select by Recording")
        self.select_by_recording_radio.setChecked(True)
        self.select_by_places_radio = QRadioButton("Select by Places")
        self.selection_mode_group.addButton(self.select_by_recording_radio, 0)
        self.selection_mode_group.addButton(self.select_by_places_radio, 1)
        self.select_by_recording_radio.toggled.connect(self.toggle_selection_mode)
        self.select_by_places_radio.toggled.connect(self.toggle_selection_mode)
        selection_mode_layout.addWidget(self.select_by_recording_radio)
        selection_mode_layout.addWidget(self.select_by_places_radio)
        recordings_layout.addLayout(selection_mode_layout)
        
        # Recording selection buttons
        recording_buttons_layout = QHBoxLayout()
        self.select_all_recordings_btn = QPushButton("Select All")
        self.select_all_recordings_btn.clicked.connect(self.select_all_recordings)
        self.select_none_recordings_btn = QPushButton("Select None")
        self.select_none_recordings_btn.clicked.connect(self.select_none_recordings)
        recording_buttons_layout.addWidget(self.select_all_recordings_btn)
        recording_buttons_layout.addWidget(self.select_none_recordings_btn)
        recordings_layout.addLayout(recording_buttons_layout)
        
        # Recordings list
        self.recordings_list = QListWidget()
        self.recordings_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.recordings_list.setMaximumHeight(200)
        self.recordings_list.setEnabled(True)
        self.populate_recordings_list()
        self.recordings_list.item(RECORDINGS_LIST.index("rosbag2_2025_07_07-12_38_45")).setSelected(True)
        recordings_layout.addWidget(self.recordings_list)
        
        # Places list (initially hidden)
        self.places_list = QListWidget()
        self.places_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.places_list.setMaximumHeight(200)
        self.places_list.setEnabled(True)
        self.places_list.itemSelectionChanged.connect(self.on_places_selection_changed)
        self.populate_places_list()
        recordings_layout.addWidget(self.places_list)
        self.places_list.setVisible(False)
        
        recordings_group.setLayout(recordings_layout)
        scroll_layout.addWidget(recordings_group)
        
        # Include columns group
        columns_group = QGroupBox("Include Columns")
        columns_layout = QVBoxLayout()
        
        # Feature set dropdown
        feature_set_layout = QHBoxLayout()
        feature_set_layout.addWidget(QLabel("Feature Set:"))
        self.feature_set_dropdown = QComboBox()
        self.feature_set_dropdown.addItems(list(FEATURE_SET_DIC.keys()))
        self.feature_set_dropdown.setCurrentText("D3 (mask + box + vitpose)")
        feature_set_layout.addWidget(self.feature_set_dropdown)
        feature_set_layout.addStretch()
        columns_layout.addLayout(feature_set_layout)
        
        columns_group.setLayout(columns_layout)
        scroll_layout.addWidget(columns_group)
        
        # Dataset parameters group
        params_group = QGroupBox("Dataset Parameters")
        params_layout = QFormLayout()
        
        # Positive cutoff
        positive_cutoff_layout = QHBoxLayout()
        self.positive_cutoff_slider = QSlider(Qt.Orientation.Horizontal)
        self.positive_cutoff_slider.setRange(0, 90)
        self.positive_cutoff_slider.setValue(15)
        self.positive_cutoff_label = QLabel("15 frames")
        self.positive_cutoff_slider.valueChanged.connect(
            lambda v: self.positive_cutoff_label.setText(str(v) + " frames")
        )
        positive_cutoff_layout.addWidget(self.positive_cutoff_slider)
        positive_cutoff_layout.addWidget(self.positive_cutoff_label)
        params_layout.addRow("T-POS:", positive_cutoff_layout)
        
        # Interaction cutoff
        interaction_cutoff_layout = QHBoxLayout()
        self.interaction_cutoff_slider = QSlider(Qt.Orientation.Horizontal)
        self.interaction_cutoff_slider.setRange(0, 90)
        self.interaction_cutoff_slider.setValue(5)
        self.interaction_cutoff_label = QLabel("5 frames")
        self.interaction_cutoff_slider.valueChanged.connect(
            lambda v: self.interaction_cutoff_label.setText(str(v) + " frames")
        )
        interaction_cutoff_layout.addWidget(self.interaction_cutoff_slider)
        interaction_cutoff_layout.addWidget(self.interaction_cutoff_label)
        params_layout.addRow("T-CUT:", interaction_cutoff_layout)
        
        # Fixed input length
        self.fixed_input_length_checkbox = QCheckBox("Fixed Input Length")
        self.fixed_input_length_checkbox.setChecked(True)
        self.fixed_input_length_checkbox.toggled.connect(self.toggle_fixed_input_length)
        params_layout.addRow("", self.fixed_input_length_checkbox)
        
        # Input length in frames
        self.input_length_spinbox = QSpinBox()
        self.input_length_spinbox.setRange(1, 1000)
        self.input_length_spinbox.setValue(30)
        params_layout.addRow("Input Length:", self.input_length_spinbox)
        
        # Stride (subsample frames)
        self.stride_spinbox = QSpinBox()
        self.stride_spinbox.setRange(1, 100)
        self.stride_spinbox.setValue(1)
        params_layout.addRow("Subsample:", self.stride_spinbox)
        
        # Min length in frames
        self.min_length_spinbox = QSpinBox()
        self.min_length_spinbox.setRange(1, 1000)
        self.min_length_spinbox.setValue(5)
        self.min_length_spinbox.setEnabled(False)
        params_layout.addRow("Min Length:", self.min_length_spinbox)
        
        # Max length in frames
        self.max_length_spinbox = QSpinBox()
        self.max_length_spinbox.setRange(1, 1000)
        self.max_length_spinbox.setValue(50)
        self.max_length_spinbox.setEnabled(False)
        params_layout.addRow("Max Length:", self.max_length_spinbox)
        
        # Inputs per track stride
        self.inputs_per_track_stride_spinbox = QSpinBox()
        self.inputs_per_track_stride_spinbox.setRange(-1, 15)
        self.inputs_per_track_stride_spinbox.setValue(-1)
        self.inputs_per_track_stride_spinbox.valueChanged.connect(self.on_inputs_per_track_stride_changed)
        params_layout.addRow("Inputs Per Track Stride:", self.inputs_per_track_stride_spinbox)
        
        # Min keypoints filter
        min_keypoints_layout = QHBoxLayout()
        self.min_keypoints_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_keypoints_slider.setRange(1, 17)
        self.min_keypoints_slider.setValue(9)
        self.min_keypoints_label = QLabel("9 valid kpts / 17")
        self.min_keypoints_slider.valueChanged.connect(
            lambda v: self.min_keypoints_label.setText(str(v) + " valid kpts / 17")
        )
        min_keypoints_layout.addWidget(self.min_keypoints_slider)
        min_keypoints_layout.addWidget(self.min_keypoints_label)
        params_layout.addRow("Filter:", min_keypoints_layout)
        
        # Normalize input
        self.normalize_in_image_checkbox = QCheckBox("Normalize (in image)")
        self.normalize_in_image_checkbox.setChecked(True)
        self.normalize_in_image_checkbox.toggled.connect(self.toggle_normalize_keypoints_in_box)
        self.normalize_in_image_checkbox.toggled.connect(self.toggle_normalize_keypoints_in_track)
        params_layout.addRow("", self.normalize_in_image_checkbox)
        
        # Normalize keypoints in box
        self.normalize_keypoints_in_box_checkbox = QCheckBox("Normalize (in box)")
        self.normalize_keypoints_in_box_checkbox.setChecked(True)
        self.normalize_keypoints_in_box_checkbox.toggled.connect(self.on_normalize_keypoints_in_box_toggled)
        params_layout.addRow("", self.normalize_keypoints_in_box_checkbox)
        
        # Normalize keypoints in track
        normalize_track_layout = QHBoxLayout()
        normalize_track_label = QLabel("Normalize (in track):")
        self.normalize_keypoints_in_track_dropdown = QComboBox()
        self.normalize_keypoints_in_track_dropdown.addItems(["none", "norm_xy", "norm_x", "norm_y"])
        self.normalize_keypoints_in_track_dropdown.setCurrentText("none")
        self.normalize_keypoints_in_track_dropdown.currentTextChanged.connect(self.on_normalize_keypoints_in_track_changed)
        normalize_track_layout.addWidget(normalize_track_label)
        normalize_track_layout.addWidget(self.normalize_keypoints_in_track_dropdown)
        params_layout.addRow("", normalize_track_layout)
        
        # Force positive samples
        self.force_positive_samples_checkbox = QCheckBox("Force Positives")
        self.force_positive_samples_checkbox.setChecked(True)
        params_layout.addRow("", self.force_positive_samples_checkbox)
        
        # Align negative tracks to biggest mask size
        self.neg_filtering = QCheckBox("Onset alignment (biggest)")
        self.neg_filtering.setChecked(True)
        params_layout.addRow("", self.neg_filtering)

        # Fix index per track
        self.fix_index_per_track_checkbox = QCheckBox("Fix Index Per Track")
        self.fix_index_per_track_checkbox.setChecked(True)
        params_layout.addRow("", self.fix_index_per_track_checkbox)
        
        # Fix index per track list
        self.fix_index_per_track_list_checkbox = QCheckBox("Fix Index Per Track List")
        self.fix_index_per_track_list_checkbox.setChecked(True)
        params_layout.addRow("", self.fix_index_per_track_list_checkbox)

        # Do recentering
        self.do_recentering_checkbox = QCheckBox("Do Recentering")
        self.do_recentering_checkbox.setChecked(False)
        params_layout.addRow("", self.do_recentering_checkbox)

        # Do recenter interaction zone
        self.do_recenter_interaction_zone_checkbox = QCheckBox("Do Recenter Interaction Zone")
        self.do_recenter_interaction_zone_checkbox.setChecked(False)
        params_layout.addRow("", self.do_recenter_interaction_zone_checkbox)

        # Center on onset
        self.center_on_onset_checkbox = QCheckBox("Center on Onset")
        self.center_on_onset_checkbox.setChecked(False)
        params_layout.addRow("", self.center_on_onset_checkbox)

        # Random flip horizontal
        self.random_flip_horizontal_checkbox = QCheckBox("Random Flip Horizontal")
        self.random_flip_horizontal_checkbox.setChecked(False)
        params_layout.addRow("", self.random_flip_horizontal_checkbox)

        # Random jitter position
        jitter_layout = QHBoxLayout()
        jitter_layout.addWidget(QLabel("Jitter X:"))
        self.jitter_x_spinbox = QDoubleSpinBox()
        self.jitter_x_spinbox.setRange(0.0, 1.0)
        self.jitter_x_spinbox.setSingleStep(0.01)
        self.jitter_x_spinbox.setValue(0.0)
        self.jitter_x_spinbox.setDecimals(3)
        jitter_layout.addWidget(self.jitter_x_spinbox)
        jitter_layout.addWidget(QLabel("Y:"))
        self.jitter_y_spinbox = QDoubleSpinBox()
        self.jitter_y_spinbox.setRange(0.0, 1.0)
        self.jitter_y_spinbox.setSingleStep(0.01)
        self.jitter_y_spinbox.setValue(0.0)
        self.jitter_y_spinbox.setDecimals(3)
        jitter_layout.addWidget(self.jitter_y_spinbox)
        params_layout.addRow("Jitter Position:", jitter_layout)

        # Return images
        self.return_images_checkbox = QCheckBox("Return Images")
        self.return_images_checkbox.setChecked(False)
        params_layout.addRow("", self.return_images_checkbox)

        # Return masks
        self.return_masks_checkbox = QCheckBox("Return Masks")
        self.return_masks_checkbox.setChecked(False)
        params_layout.addRow("", self.return_masks_checkbox)

        # Standardization dropdown
        self.use_standardization_dropdown = QComboBox()
        self.use_standardization_dropdown.addItems(["all", "none", "mask_only"])
        self.use_standardization_dropdown.setCurrentText("all")
        params_layout.addRow("Standardize:", self.use_standardization_dropdown)
        
        # Dataset revision dropdown
        revision_layout = QHBoxLayout()
        revision_label = QLabel("Dataset Revision:")
        self.hf_dataset_revision_dropdown = QComboBox()
        self.hf_dataset_revision_dropdown.addItems(["legacy", "main"])
        self.hf_dataset_revision_dropdown.setCurrentText("main")
        revision_layout.addWidget(revision_label)
        revision_layout.addWidget(self.hf_dataset_revision_dropdown)
        params_layout.addRow("", revision_layout)
        
        # Initialize inputs_per_track_stride state (ensure checkboxes are in correct state)
        self.on_inputs_per_track_stride_changed(self.inputs_per_track_stride_spinbox.value())
                
        params_group.setLayout(params_layout)
        scroll_layout.addWidget(params_group)
        
        # Dataset creation buttons
        buttons_layout = QVBoxLayout()
        
        self.load_checkpoint_btn = QPushButton("Load Config from Checkpoint")
        self.load_checkpoint_btn.clicked.connect(self.load_config_from_checkpoint)
        
        self.load_config_and_model_btn = QPushButton("Load Config And Model from Checkpoint")
        self.load_config_and_model_btn.clicked.connect(self.load_config_and_model_from_checkpoint)
        
        # self.validate_btn = QPushButton("Validate Config")
        # self.validate_btn.clicked.connect(self.validate_config)
        
        self.create_dataset_btn = QPushButton("Create Dataset")
        self.create_dataset_btn.clicked.connect(self.create_dataset)
        
        buttons_layout.addWidget(self.load_checkpoint_btn)
        buttons_layout.addWidget(self.load_config_and_model_btn)
        # buttons_layout.addWidget(self.validate_btn)
        buttons_layout.addWidget(self.create_dataset_btn)
        
        scroll_layout.addLayout(buttons_layout)
        scroll_layout.addStretch()
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        
        layout.addWidget(scroll)
        self.setLayout(layout)
    
    def setup_connections(self):
        """Setup signal connections"""
        pass
    
    def populate_recordings_list(self):
        """Populate the recordings list with available recordings"""
        for recording in RECORDINGS_LIST:
            item = QListWidgetItem(recording)
            item.setSelected(True)  # Select all by default
            self.recordings_list.addItem(item)
    
    def populate_places_list(self):
        """Populate the places list with available places"""
        for place in sorted(UNIQUE_PLACES_RECORDINGS.keys()):
            item = QListWidgetItem(place)
            item.setSelected(False)
            self.places_list.addItem(item)
    
    def toggle_selection_mode(self, checked):
        """Toggle between recording and places selection modes"""
        if not checked:
            return  # Only handle when the radio button is checked
        
        is_recording_mode = self.select_by_recording_radio.isChecked()
        
        # Show/hide appropriate lists
        self.recordings_list.setVisible(is_recording_mode)
        self.places_list.setVisible(not is_recording_mode)
        
        # Enable/disable appropriate controls
        self.select_all_recordings_btn.setEnabled(is_recording_mode)
        self.select_none_recordings_btn.setEnabled(is_recording_mode)
        
        # Update recordings based on places selection when switching to places mode
        if not is_recording_mode:
            self.on_places_selection_changed()
    
    def on_places_selection_changed(self):
        """Handle place selection changes - automatically select all recordings for selected places"""
        # First, clear all recording selections
        for i in range(self.recordings_list.count()):
            self.recordings_list.item(i).setSelected(False)
        
        # Get selected places
        selected_places = []
        for i in range(self.places_list.count()):
            if self.places_list.item(i).isSelected():
                selected_places.append(self.places_list.item(i).text())
        
        # Select all recordings for selected places
        for place in selected_places:
            if place in UNIQUE_PLACES_RECORDINGS:
                recordings_for_place = UNIQUE_PLACES_RECORDINGS[place]
                for recording in recordings_for_place:
                    # Find and select the recording in the recordings list
                    for i in range(self.recordings_list.count()):
                        if self.recordings_list.item(i).text() == recording:
                            self.recordings_list.item(i).setSelected(True)
                            break
    
    def select_all_recordings(self):
        """Select all recordings"""
        for i in range(self.recordings_list.count()):
            self.recordings_list.item(i).setSelected(True)
    
    def select_none_recordings(self):
        """Select no recordings"""
        for i in range(self.recordings_list.count()):
            self.recordings_list.item(i).setSelected(False)
    
    def toggle_fixed_input_length(self, checked):
        """Toggle fixed input length mode"""
        self.input_length_spinbox.setEnabled(checked)
        self.min_length_spinbox.setEnabled(not checked)
        self.max_length_spinbox.setEnabled(not checked)
    
    def on_inputs_per_track_stride_changed(self, value):
        """Handle inputs_per_track_stride spinbox change - disable fix index checkboxes when >= 0"""
        if value >= 0:
            # When stride is >= 0, uncheck and disable both fix index checkboxes
            self.fix_index_per_track_checkbox.setChecked(False)
            self.fix_index_per_track_checkbox.setEnabled(False)
            self.fix_index_per_track_list_checkbox.setChecked(False)
            self.fix_index_per_track_list_checkbox.setEnabled(False)
        else:
            # When stride is -1, enable both checkboxes
            self.fix_index_per_track_checkbox.setEnabled(True)
            self.fix_index_per_track_list_checkbox.setEnabled(True)
    
    def toggle_normalize_keypoints_in_box(self, checked):
        """Toggle normalize keypoints in box based on normalize input state"""
        self.normalize_keypoints_in_box_checkbox.setEnabled(checked)
        if not checked:
            # If normalize input is unchecked, also uncheck normalize keypoints in box
            self.normalize_keypoints_in_box_checkbox.setChecked(False)
    
    def toggle_normalize_keypoints_in_track(self, checked):
        """Toggle normalize keypoints in track dropdown based on normalize input state"""
        self.normalize_keypoints_in_track_dropdown.setEnabled(checked)
        if not checked:
            # If normalize input is unchecked, set normalize_keypoints_in_track to "none"
            self.normalize_keypoints_in_track_dropdown.setCurrentText("none")
    
    def on_normalize_keypoints_in_box_toggled(self, checked):
        """Handle normalize_keypoints_in_box checkbox toggle - enforce mutual exclusivity"""
        if checked:
            # If normalize_keypoints_in_box is checked, set normalize_keypoints_in_track to "none" (mutual exclusivity)
            self.normalize_keypoints_in_track_dropdown.setCurrentText("none")
    
    def on_normalize_keypoints_in_track_changed(self, value):
        """Handle normalize_keypoints_in_track dropdown change - enforce mutual exclusivity"""
        if value != "none":
            # If normalize_keypoints_in_track is not "none", uncheck normalize_keypoints_in_box (mutual exclusivity)
            self.normalize_keypoints_in_box_checkbox.setChecked(False)
    
    def get_include_columns(self):
        """Get the selected columns from the feature set dropdown"""
        selected_feature_set = self.feature_set_dropdown.currentText()
        feature_set = FEATURE_SET_DIC[selected_feature_set]
        
        # If feature set is "all", return "all"
        if feature_set == "all":
            return "all"
        
        # Otherwise return the feature set list
        return feature_set
    
    def get_include_recordings(self):
        """Get the selected recordings"""
        # Get selected recordings from the list
        selected_recordings = []
        for i in range(self.recordings_list.count()):
            if self.recordings_list.item(i).isSelected():
                selected_recordings.append(self.recordings_list.item(i).text())
        return selected_recordings if selected_recordings else "all"
    
    def get_config(self):
        """Get the current configuration"""
        config = {
            "include_recordings": self.get_include_recordings(),
            "include_tracks": "all",
            "include_columns": self.get_include_columns(),
            "positive_cutoff": self.positive_cutoff_slider.value(),
            "interaction_cutoff": self.interaction_cutoff_slider.value(),
            "fixed_input_length": self.fixed_input_length_checkbox.isChecked(),
            "input_length_in_frames": self.input_length_spinbox.value() if self.fixed_input_length_checkbox.isChecked() else None,
            "min_length_in_frames": self.min_length_spinbox.value() if not self.fixed_input_length_checkbox.isChecked() else None,
            "max_length_in_frames": self.max_length_spinbox.value() if not self.fixed_input_length_checkbox.isChecked() else None,
            "subsample_frames": self.stride_spinbox.value(),
            "inputs_per_track_stride": self.inputs_per_track_stride_spinbox.value(),
            "min_keypoints_filter": self.min_keypoints_slider.value(),
            "additional_filtering_dict": {"mask_size": {"min": 1000, "max": 1e7}},
            "return_images": self.return_images_checkbox.isChecked(),
            "return_masks": self.return_masks_checkbox.isChecked(),
            "normalize_in_image": self.normalize_in_image_checkbox.isChecked(),
            "normalize_keypoints_in_box": self.normalize_keypoints_in_box_checkbox.isChecked(),
            "normalize_keypoints_in_track": self.normalize_keypoints_in_track_dropdown.currentText(),
            "force_positive_samples": self.force_positive_samples_checkbox.isChecked(),
            "force_aligment_with_biggest_mask_size": self.neg_filtering.isChecked(),
            "standardize_data": self.use_standardization_dropdown.currentText(),
            "fix_index_per_track": self.fix_index_per_track_checkbox.isChecked(),
            "fix_index_per_track_list": [3662, 5427, 3606, 3726, 3417, 6031, 7527, 1501, 4501, 9588, 2712, 4509, 2752, 57, 9256, 3417, 8694, 9336, 6870, 3587, 2675, 3613, 9281, 4883, 7570, 8967, 1654, 5194, 9746, 4310, 2848, 9954] if self.fix_index_per_track_list_checkbox.isChecked() else None, # to fully fix the index choice from a repetition to another
            "do_recentering": self.do_recentering_checkbox.isChecked(),
            "do_recenter_interaction_zone": self.do_recenter_interaction_zone_checkbox.isChecked(),
            "center_on_onset": self.center_on_onset_checkbox.isChecked(),
            "random_flip_horizontal": self.random_flip_horizontal_checkbox.isChecked(),
            "random_jitter_position": (self.jitter_x_spinbox.value(), self.jitter_y_spinbox.value()),
            "hf_dataset_revision": "3c8a342548534b6b92d32b0099e266962facdf45" if self.hf_dataset_revision_dropdown.currentText() == "legacy" else self.hf_dataset_revision_dropdown.currentText(),
        }
        return config
    
    def validate_config(self):
        """Validate the current configuration"""
        config = self.get_config()
        
        # Basic validation
        if config["fixed_input_length"]:
            if config["input_length_in_frames"] is None:
                raise ValueError("Input length must be specified when fixed_input_length is True")
            if config["input_length_in_frames"] % config["subsample_frames"] != 0:
                raise ValueError("Input length must be divisible by stride (subsample_frames)")
            if config["input_length_in_frames"] < config["subsample_frames"]:
                raise ValueError("Input length must be greater than stride (subsample_frames)")
        else:
            if config["min_length_in_frames"] is None or config["max_length_in_frames"] is None:
                raise ValueError("Min and max length must be specified when fixed_input_length is False")
            if config["min_length_in_frames"] > config["max_length_in_frames"]:
                raise ValueError("Min length must be less than or equal to max length")
        
        if len(config["include_columns"]) == 0:
            raise ValueError("At least one column must be selected")
        
        # Recording validation
        if config["include_recordings"] != "all" and len(config["include_recordings"]) == 0:
            raise ValueError("At least one recording must be selected")
        
        # Normalize keypoints in box validation
        if config["normalize_keypoints_in_box"] and not config["normalize_in_image"]:
            raise ValueError("normalize_keypoints_in_box requires normalize_in_image to be True")
        
        # Normalize keypoints in track validation
        if config["normalize_keypoints_in_track"] not in ["none", "norm_xy", "norm_x", "norm_y"]:
            raise ValueError("normalize_keypoints_in_track must be 'none', 'norm_xy', 'norm_x', or 'norm_y'")
        if config["normalize_keypoints_in_track"] != "none" and not config["normalize_in_image"]:
            raise ValueError("normalize_keypoints_in_track requires normalize_in_image to be True")
        
        # Mutual exclusivity validation
        if config["normalize_keypoints_in_box"] and config["normalize_keypoints_in_track"] != "none":
            raise ValueError("normalize_keypoints_in_box and normalize_keypoints_in_track cannot be enabled at the same time")
        
        QMessageBox.information(self, "Validation", "Configuration is valid!")
        
    
    def load_config_from_checkpoint(self):
        """Load configuration from a model checkpoint file"""
        return_message_non_critical_elements_missing = []
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Model Checkpoint", 
                here, 
                "PyTorch Model (*.pth *.pt)"
            )
            if file_path is None or file_path == "":
                return
            
            # Load checkpoint
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
            
            # Check if it's an RF model
            if 'config' not in checkpoint:
                if 'hyperparameters' in checkpoint:
                    checkpoint["config"] = checkpoint["hyperparameters"]
                prError("Checkpoint does not contain 'config' or 'hyperparameters' key. Cannot load config.")
                QMessageBox.warning(
                    self, 
                    "Load Config", 
                    "Checkpoint does not contain hyperparameters. Cannot load config."
                )
                return
            
            config = checkpoint["config"]
            config = update_old_config_dict(config)
            
            # Check if it's an RF model
            model_type = config["force_model_type"].lower()
            if model_type == "rf":
                QMessageBox.warning(
                    self, 
                    "Load Config", 
                    "RF models are not supported. Please select a non-RF model checkpoint."
                )
                return
            
            # Update UI controls based on config
            # Positive cutoff (use train value if available, otherwise val)
            positive_cutoff = config["positive_cutoff_val"]
            if positive_cutoff is not None:
                self.positive_cutoff_slider.setValue(int(positive_cutoff))
                self.positive_cutoff_label.setText(f"{int(positive_cutoff)} frames")
            
            # Interaction cutoff (use train value if available, otherwise val)
            interaction_cutoff = config["interaction_cutoff_val"]
            if interaction_cutoff is not None:
                self.interaction_cutoff_slider.setValue(int(interaction_cutoff))
                self.interaction_cutoff_label.setText(f"{int(interaction_cutoff)} frames")
            
            # Fixed input length
            fixed_input_length = config["fixed_input_length"]
            self.fixed_input_length_checkbox.setChecked(fixed_input_length)
            self.toggle_fixed_input_length(fixed_input_length)
            
            # Input length in frames
            input_length = config["input_length_in_frames"]
            if input_length is not None:
                self.input_length_spinbox.setValue(int(input_length))
            
            # Subsample frames
            subsample_frames = config["subsample_frames"]
            if subsample_frames is not None:
                self.stride_spinbox.setValue(int(subsample_frames))
            
            # Min/Max length in frames
            min_length = config["min_length_in_frames"]
            if min_length is not None:
                self.min_length_spinbox.setValue(int(min_length))
            
            max_length = config["max_length_in_frames"]
            if max_length is not None:
                self.max_length_spinbox.setValue(int(max_length))
            
            # Inputs per track stride
            inputs_per_track_stride = config["inputs_per_track_stride_val"]
            if inputs_per_track_stride is not None:
                self.inputs_per_track_stride_spinbox.setValue(int(inputs_per_track_stride))
            
            # Min keypoints filter
            min_keypoints_filter = config["min_keypoints_filter"]
            if min_keypoints_filter is not None:
                self.min_keypoints_slider.setValue(int(min_keypoints_filter))
                self.min_keypoints_label.setText(f"{int(min_keypoints_filter)} valid kpts / 17")
            
            # Normalize in image
            normalize_in_image = config["normalize_in_image"]
            self.normalize_in_image_checkbox.setChecked(normalize_in_image)
            self.toggle_normalize_keypoints_in_box(normalize_in_image)
            self.toggle_normalize_keypoints_in_track(normalize_in_image)
            
            # Normalize keypoints in box
            normalize_keypoints_in_box = config["normalize_keypoints_in_box"]
            self.normalize_keypoints_in_box_checkbox.setChecked(normalize_keypoints_in_box)
            
            # Normalize keypoints in track
            normalize_keypoints_in_track = config["normalize_keypoints_in_track"]
            # Handle backward compatibility: convert boolean to string
            if isinstance(normalize_keypoints_in_track, bool):
                normalize_keypoints_in_track = "norm_xy" if normalize_keypoints_in_track else "none"
            elif normalize_keypoints_in_track not in ["none", "norm_xy", "norm_x", "norm_y"]:
                normalize_keypoints_in_track = "none"  # Default to "none" if invalid value
            self.normalize_keypoints_in_track_dropdown.setCurrentText(normalize_keypoints_in_track)
            
            # Force positive samples
            force_positive_samples = config["force_positive_samples"]
            self.force_positive_samples_checkbox.setChecked(force_positive_samples)
            
            # Force alignment with biggest mask size (use train value if available, otherwise val)
            force_alignment = config["force_aligment_with_biggest_mask_size_val"]
            if force_alignment is not None:
                self.neg_filtering.setChecked(force_alignment)
            
            # Fix index per track (use train value if available, otherwise val)
            fix_index_per_track = config["fix_index_per_track_val"]
            if fix_index_per_track is not None:
                self.fix_index_per_track_checkbox.setChecked(fix_index_per_track)
            
            # Fix index per track list (use train value if available, otherwise val)
            fix_index_per_track_list = config["fix_index_per_track_list_val"]
            if fix_index_per_track_list is not None:
                self.fix_index_per_track_list_checkbox.setChecked(fix_index_per_track_list is not None)
            
            # Standardize data
            standardize_data = config["standardize_data"]
            if standardize_data in ["all", "none", "mask_only"]:
                self.use_standardization_dropdown.setCurrentText(standardize_data)
            
            # Do recentering
            do_recentering = config["do_recentering_val"]
            self.do_recentering_checkbox.setChecked(do_recentering)

            # Do recenter interaction zone
            if not "do_recenter_interaction_zone" in config:
                return_message_non_critical_elements_missing.append(("do_recenter_interaction_zone",False))
                do_recenter_interaction_zone = False
            else:
                do_recenter_interaction_zone = config["do_recenter_interaction_zone"]
            self.do_recenter_interaction_zone_checkbox.setChecked(do_recenter_interaction_zone)

            # Center on onset
            center_on_onset = config["center_on_onset_val"]
            self.center_on_onset_checkbox.setChecked(center_on_onset)

            # Random flip horizontal
            random_flip_horizontal = config["random_flip_horizontal_val"]
            self.random_flip_horizontal_checkbox.setChecked(random_flip_horizontal)

            # Random jitter position
            random_jitter_position = config["random_jitter_position_val"]
            self.jitter_x_spinbox.setValue(float(random_jitter_position[0]))
            self.jitter_y_spinbox.setValue(float(random_jitter_position[1]))

            # Return images
            if not "return_images" in config:
                return_message_non_critical_elements_missing.append(("return_images",False))
                return_images = False
            else:
                return_images = config["return_images"]
            self.return_images_checkbox.setChecked(return_images)

            # Return masks
            if not "return_masks" in config:
                return_message_non_critical_elements_missing.append(("return_masks",False))
                return_masks = False
            else:
                return_masks = config["return_masks"]
            self.return_masks_checkbox.setChecked(return_masks)
            
            # Feature set / include_columns
            include_columns = config["include_columns"]
            if include_columns is not None:
                # Check if it matches one of the existing feature sets
                matched_feature_set = None
                for feature_set_name, feature_set_columns in FEATURE_SET_DIC.items():
                    # Skip "Custom from checkpoint" when matching
                    if feature_set_name == "Custom from checkpoint":
                        continue
                    # Handle "all" case
                    if include_columns == "all" and feature_set_columns == "all":
                        matched_feature_set = feature_set_name
                        break
                    # Handle list comparison
                    if isinstance(include_columns, list) and isinstance(feature_set_columns, list):
                        # Compare sets to ignore order
                        if set(include_columns) == set(feature_set_columns):
                            matched_feature_set = feature_set_name
                            break
                
                if matched_feature_set:
                    # Set to existing feature set
                    self.feature_set_dropdown.setCurrentText(matched_feature_set)
                    # Remove "Custom from checkpoint" if it exists
                    for i in range(self.feature_set_dropdown.count()):
                        if self.feature_set_dropdown.itemText(i) == "Custom from checkpoint":
                            self.feature_set_dropdown.removeItem(i)
                            break
                    # Remove from dictionary if it was there
                    if "Custom from checkpoint" in FEATURE_SET_DIC:
                        del FEATURE_SET_DIC["Custom from checkpoint"]
                else:
                    # Add custom feature set option if it doesn't exist
                    custom_exists = False
                    for i in range(self.feature_set_dropdown.count()):
                        if self.feature_set_dropdown.itemText(i) == "Custom from checkpoint":
                            custom_exists = True
                            break
                    
                    if not custom_exists:
                        self.feature_set_dropdown.addItem("Custom from checkpoint")
                    
                    # Store/update the custom columns in the dictionary
                    FEATURE_SET_DIC["Custom from checkpoint"] = include_columns
                    self.feature_set_dropdown.setCurrentText("Custom from checkpoint")
            
            # Include recordings (use train value if available, otherwise val)
            include_recordings = config["include_recordings_val"]
            if include_recordings is not None and include_recordings != "all":
                # Clear all selections first
                for i in range(self.recordings_list.count()):
                    self.recordings_list.item(i).setSelected(False)
                
                # Select recordings from checkpoint
                if isinstance(include_recordings, list):
                    for recording in include_recordings:
                        for i in range(self.recordings_list.count()):
                            if self.recordings_list.item(i).text() == recording:
                                self.recordings_list.item(i).setSelected(True)
                                break
            
            # Unfreeze controls if they were frozen (since we're not loading a model)
            if self.model_loaded:
                self.model_loaded = False
                self.freeze_model_dependent_controls(freeze=False)
            
            missing_elements_message = ""
            if return_message_non_critical_elements_missing:
                missing_elements_message = "\n\nNon-critical keys missing from config:\n"
                for element, value in return_message_non_critical_elements_missing:
                    missing_elements_message += f"{element}: used default value of {value}\n"
            
            QMessageBox.information(
                self, 
                "Load Config", 
                f"Configuration loaded from checkpoint:\n{file_path}{missing_elements_message}"
            )
            
        except Exception as e:
            prError(f"Failed to load config from checkpoint: {e}")
            QMessageBox.critical(self, "Load Config Error", str(e))
    
    def freeze_model_dependent_controls(self, freeze=True):
        """Freeze or unfreeze controls that should not be modified when model is loaded"""
        self.feature_set_dropdown.setEnabled(not freeze)
        self.use_standardization_dropdown.setEnabled(not freeze)
        self.normalize_in_image_checkbox.setEnabled(not freeze)
        self.normalize_keypoints_in_box_checkbox.setEnabled(not freeze)
        self.normalize_keypoints_in_track_dropdown.setEnabled(not freeze)
    
    def load_config_and_model_from_checkpoint(self):
        """Load configuration and model from a checkpoint file"""
        return_message_non_critical_elements_missing = []
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Model Checkpoint", 
                os.path.join(here, "experiments", "results"), 
                "PyTorch Model (*.pth *.pt)"
            )
            if file_path is None or file_path == "":
                return
            
            # Load checkpoint
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
            
            # Check if it's an RF model
            if 'config' not in checkpoint:
                if 'hyperparameters' in checkpoint:
                    checkpoint["config"] = checkpoint["hyperparameters"]
                prError("Checkpoint does not contain 'config' or 'hyperparameters' key. Cannot load config.")
                QMessageBox.warning(
                    self, 
                    "Load Config And Model", 
                    "Checkpoint does not contain hyperparameters. Cannot load config."
                )
                return
            
            config = checkpoint["config"]
            config = update_old_config_dict(config)
            
            # Check if it's an RF model
            model_type = config["force_model_type"].lower()
            if model_type == "rf":
                QMessageBox.warning(
                    self, 
                    "Load Config And Model", 
                    "RF models are not supported. Please select a non-RF model checkpoint."
                )
                return
            
            # First load the config (reuse existing method)
            # We'll manually update controls since we need to do model loading too
            # But let's call the config loading logic inline
            
            # Update UI controls based on config (same as load_config_from_checkpoint)
            # Positive cutoff
            positive_cutoff = config["positive_cutoff_val"]
            if positive_cutoff is not None:
                self.positive_cutoff_slider.setValue(int(positive_cutoff))
                self.positive_cutoff_label.setText(f"{int(positive_cutoff)} frames")
            
            # Interaction cutoff
            interaction_cutoff = config["interaction_cutoff_val"]
            if interaction_cutoff is not None:
                self.interaction_cutoff_slider.setValue(int(interaction_cutoff))
                self.interaction_cutoff_label.setText(f"{int(interaction_cutoff)} frames")
            
            # Fixed input length
            fixed_input_length = config["fixed_input_length"]
            self.fixed_input_length_checkbox.setChecked(fixed_input_length)
            self.toggle_fixed_input_length(fixed_input_length)
            
            # Input length in frames
            input_length = config["input_length_in_frames"]
            if input_length is not None:
                self.input_length_spinbox.setValue(int(input_length))
            
            # Subsample frames
            subsample_frames = config["subsample_frames"]
            if subsample_frames is not None:
                self.stride_spinbox.setValue(int(subsample_frames))
            
            # Min/Max length
            min_length = config["min_length_in_frames"]
            if min_length is not None:
                self.min_length_spinbox.setValue(int(min_length))
            
            max_length = config["max_length_in_frames"]
            if max_length is not None:
                self.max_length_spinbox.setValue(int(max_length))
            
            # Inputs per track stride
            inputs_per_track_stride = config["inputs_per_track_stride_val"]
            if inputs_per_track_stride is not None:
                self.inputs_per_track_stride_spinbox.setValue(int(inputs_per_track_stride))
            
            # Min keypoints filter
            min_keypoints_filter = config["min_keypoints_filter"]
            if min_keypoints_filter is not None:
                self.min_keypoints_slider.setValue(int(min_keypoints_filter))
                self.min_keypoints_label.setText(f"{int(min_keypoints_filter)} valid kpts / 17")
            
            # Normalize in image
            normalize_in_image = config["normalize_in_image"]
            self.normalize_in_image_checkbox.setChecked(normalize_in_image)
            self.toggle_normalize_keypoints_in_box(normalize_in_image)
            self.toggle_normalize_keypoints_in_track(normalize_in_image)
            
            # Normalize keypoints in box
            normalize_keypoints_in_box = config["normalize_keypoints_in_box"]
            self.normalize_keypoints_in_box_checkbox.setChecked(normalize_keypoints_in_box)
            
            # Normalize keypoints in track
            normalize_keypoints_in_track = config["normalize_keypoints_in_track"]
            # Handle backward compatibility: convert boolean to string
            if isinstance(normalize_keypoints_in_track, bool):
                normalize_keypoints_in_track = "norm_xy" if normalize_keypoints_in_track else "none"
            elif normalize_keypoints_in_track not in ["none", "norm_xy", "norm_x", "norm_y"]:
                normalize_keypoints_in_track = "none"  # Default to "none" if invalid value
            self.normalize_keypoints_in_track_dropdown.setCurrentText(normalize_keypoints_in_track)
            
            # Force positive samples
            force_positive_samples = config["force_positive_samples"]
            self.force_positive_samples_checkbox.setChecked(force_positive_samples)
            
            # Force alignment
            force_alignment = config["force_aligment_with_biggest_mask_size_val"]
            if force_alignment is not None:
                self.neg_filtering.setChecked(force_alignment)
            
            # Fix index per track
            fix_index_per_track = config["fix_index_per_track_val"]
            if fix_index_per_track is not None:
                self.fix_index_per_track_checkbox.setChecked(fix_index_per_track)
            
            # Fix index per track list
            fix_index_per_track_list = config["fix_index_per_track_list_val"]
            if fix_index_per_track_list is not None:
                self.fix_index_per_track_list_checkbox.setChecked(fix_index_per_track_list is not None)
            
            # Standardize data
            standardize_data = config["standardize_data"]
            if standardize_data in ["all", "none", "mask_only"]:
                self.use_standardization_dropdown.setCurrentText(standardize_data)
            
            # Do recentering
            do_recentering = config["do_recentering_val"]
            self.do_recentering_checkbox.setChecked(do_recentering)

            # Do recenter interaction zone
            if not "do_recenter_interaction_zone" in config:
                return_message_non_critical_elements_missing.append(("do_recenter_interaction_zone",False))
                do_recenter_interaction_zone = False
            else:
                do_recenter_interaction_zone = config["do_recenter_interaction_zone"]
            self.do_recenter_interaction_zone_checkbox.setChecked(do_recenter_interaction_zone)

            # Center on onset
            center_on_onset = config["center_on_onset_val"]
            self.center_on_onset_checkbox.setChecked(center_on_onset)

            # Random flip horizontal
            random_flip_horizontal = config["random_flip_horizontal_val"]
            self.random_flip_horizontal_checkbox.setChecked(random_flip_horizontal)

            # Random jitter position
            random_jitter_position = config["random_jitter_position_val"]
            self.jitter_x_spinbox.setValue(float(random_jitter_position[0]))
            self.jitter_y_spinbox.setValue(float(random_jitter_position[1]))

            # Return images
            if not "return_images" in config:
                return_message_non_critical_elements_missing.append(("return_images",False))
                return_images = False
            else:
                return_images = config["return_images"]
            self.return_images_checkbox.setChecked(return_images)

            # Return masks
            if not "return_masks" in config:
                return_message_non_critical_elements_missing.append(("return_masks",False))
                return_masks = False
            else:
                return_masks = config["return_masks"]
            self.return_masks_checkbox.setChecked(return_masks)
            
            # Feature set / include_columns
            include_columns = config["include_columns"]
            if include_columns is not None:
                # Check if it matches one of the existing feature sets
                matched_feature_set = None
                for feature_set_name, feature_set_columns in FEATURE_SET_DIC.items():
                    if feature_set_name == "Custom from checkpoint":
                        continue
                    if include_columns == "all" and feature_set_columns == "all":
                        matched_feature_set = feature_set_name
                        break
                    if isinstance(include_columns, list) and isinstance(feature_set_columns, list):
                        if set(include_columns) == set(feature_set_columns):
                            matched_feature_set = feature_set_name
                            break
                
                if matched_feature_set:
                    self.feature_set_dropdown.setCurrentText(matched_feature_set)
                    for i in range(self.feature_set_dropdown.count()):
                        if self.feature_set_dropdown.itemText(i) == "Custom from checkpoint":
                            self.feature_set_dropdown.removeItem(i)
                            break
                    if "Custom from checkpoint" in FEATURE_SET_DIC:
                        del FEATURE_SET_DIC["Custom from checkpoint"]
                else:
                    custom_exists = False
                    for i in range(self.feature_set_dropdown.count()):
                        if self.feature_set_dropdown.itemText(i) == "Custom from checkpoint":
                            custom_exists = True
                            break
                    
                    if not custom_exists:
                        self.feature_set_dropdown.addItem("Custom from checkpoint")
                    
                    FEATURE_SET_DIC["Custom from checkpoint"] = include_columns
                    self.feature_set_dropdown.setCurrentText("Custom from checkpoint")
            
            # Include recordings
            include_recordings = (
                config["include_recordings_val"]
            )
            if include_recordings is not None and include_recordings != "all":
                for i in range(self.recordings_list.count()):
                    self.recordings_list.item(i).setSelected(False)
                
                if isinstance(include_recordings, list):
                    for recording in include_recordings:
                        for i in range(self.recordings_list.count()):
                            if self.recordings_list.item(i).text() == recording:
                                self.recordings_list.item(i).setSelected(True)
                                break
            
            # Now load the model
            # Get model type
            model_type = config["force_model_type"].lower()
            
            # Get input dimensions from include_columns
            if include_columns == "all":
                # Need to get from dataset if available, otherwise estimate
                # For now, we'll need to create a temporary dataset or wait for user to create one
                QMessageBox.warning(
                    self,
                    "Load Model",
                    "Cannot determine input dimensions. Please create a dataset first, then load the model again."
                )
                return

            # Calculate sequence length
            if input_length is not None and subsample_frames is not None:
                sequence_length = int(input_length) // max(1, int(subsample_frames))
            else:
                QMessageBox.warning(
                    self,
                    "Load Model",
                    "Cannot determine sequence length from checkpoint."
                )
                return
            
            # Infer input_dim from model state_dict
            # For MLP: first layer weight shape is (hidden_dim, input_dim * sequence_length)
            # For LSTM: first layer weight shape contains input_dim information
            input_dim = None
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if model_type == "mlp":
                    # Find the first linear layer (usually 'mlp.0.weight')
                    for key in state_dict.keys():
                        if 'mlp.0.weight' in key or ('weight' in key and 'mlp' in key and '0' in key):
                            weight = state_dict[key]
                            # MLP first layer: (hidden_dim, input_dim * sequence_length)
                            flattened_dim = weight.shape[1]
                            input_dim = flattened_dim // sequence_length
                            break
                elif model_type == "lstm":
                    # Find the LSTM weight_ih_l0 (input-to-hidden weights for first layer)
                    for key in state_dict.keys():
                        if 'lstm.weight_ih_l0' in key:
                            weight = state_dict[key]
                            # LSTM weight_ih shape: (4 * hidden_dim, input_dim)
                            # 4 because of input, forget, cell, output gates
                            input_dim = weight.shape[1]
                            break
                elif model_type == "stg_nf":
                    input_dim = None
            
            # Create model based on type
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            prInfo(f"Using device: {device} for inference")
            
            if model_type == "mlp":
                hidden_dims = config["hidden_dims"]
                dropout = config["dropout"]
                model = MLPInteractionPredictor(
                    input_dim=input_dim,
                    sequence_length=sequence_length,
                    hidden_dims=hidden_dims,
                    dropout=dropout
                )
            elif model_type == "lstm":
                lstm_hidden_dim = config["lstm_hidden_dim"]
                lstm_num_layers = config["lstm_num_layers"]
                lstm_dropout = config["lstm_dropout"]
                model = LSTMInteractionPredictor(
                    input_dim=input_dim,
                    sequence_length=sequence_length,
                    hidden_dim=lstm_hidden_dim,
                    num_layers=lstm_num_layers,
                    dropout=lstm_dropout,
                    bidirectional=False
                )
            elif model_type == "stg_nf":
                stg_nf_hidden_channels = config["stg_nf_hidden_channels"]
                stg_nf_K = config["stg_nf_K"]
                stg_nf_L = config["stg_nf_L"]
                stg_nf_R = config["stg_nf_R"]
                stg_nf_actnorm_scale = config["stg_nf_actnorm_scale"]
                stg_nf_edge_importance = config["stg_nf_edge_importance"]
                stg_nf_max_hops = config["stg_nf_max_hops"]
                model = STG_NF(device=device,
                                pose_shape=(2, sequence_length, 18),
                                hidden_channels=stg_nf_hidden_channels,
                                K=stg_nf_K,
                                L=stg_nf_L,
                                R=stg_nf_R,
                                actnorm_scale=stg_nf_actnorm_scale,
                                flow_permutation="permute",
                                flow_coupling="affine",
                                LU_decomposed=True,
                                learn_top=False,
                                edge_importance=stg_nf_edge_importance,
                                temporal_kernel_size=None,
                                strategy="uniform",
                                max_hops=stg_nf_max_hops,)
                model.set_actnorm_init()
                prSuccess("ActNorm2d initialized")

            elif model_type == "stgcn":
                # STGCN model loading (similar to infer.py)
                model = STGCN(
                    in_channels=config["stgcn_in_channels"],
                    num_class=1,
                    graph_args={"layout": config["stgcn_layout"], "strategy": 'spatial'},
                    edge_importance_weighting=config["stgcn_edge_importance_weighting"],
                )
                # For STGCN, input_dim is not used in the same way
                input_dim = None  # Not applicable for STGCN
            elif model_type == "skateformer":
                # SkateFormer model loading (similar to infer.py)
                assert(sequence_length % 8 == 0), f"Sequence length must be divisible by 8 for SkateFormer, got {sequence_length}"
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
                )
                # For SkateFormer, input_dim is not used in the same way
                input_dim = None  # Not applicable for SkateFormer
            else:
                QMessageBox.warning(
                    self,
                    "Load Model",
                    f"Unsupported model type: {model_type}"
                )
                return
            
            # Load model weights
            if 'model_state_dict' not in checkpoint:
                QMessageBox.warning(
                    self,
                    "Load Model",
                    "Checkpoint does not contain model_state_dict."
                )
                return
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to(device)
            
            # Store model information for export
            main_window = self.window()
            if hasattr(main_window, 'data_visualization'):
                main_window.data_visualization.model_checkpoint_path = file_path
                main_window.data_visualization.model_config = config
                main_window.data_visualization.model_type = model_type
                main_window.data_visualization.device = device  # Store device for inference
                main_window.data_visualization.set_model(model, expected_sequence_length=sequence_length)
            
            # Freeze model-dependent controls
            self.model_loaded = True
            self.freeze_model_dependent_controls(freeze=True)
            
            missing_elements_message = ""
            if return_message_non_critical_elements_missing:
                missing_elements_message = "\n\nNon-critical keys missing from config:\n"
                for element, value in return_message_non_critical_elements_missing:
                    missing_elements_message += f"{element}: used default value of {value}\n"
            
            # Extract expected validation metrics from checkpoint
            val_auc = checkpoint.get('val_auc', None)
            val_ap = checkpoint.get('val_ap', None)
            val_f1 = checkpoint.get('val_f1_with_adaptative_threshold', checkpoint.get('val_f1', None))
            epoch = checkpoint.get('epoch', None)
            
            metrics_message = ""
            if val_auc is not None or val_ap is not None:
                metrics_message = "\n\nExpected validation metrics:\n"
                if val_auc is not None:
                    metrics_message += f"  AUC: {val_auc:.4f}\n"
                if val_ap is not None:
                    metrics_message += f"  AP: {val_ap:.4f}\n"
                if val_f1 is not None:
                    metrics_message += f"  F1 (adaptive threshold): {val_f1:.4f}\n"
                if epoch is not None:
                    metrics_message += f"  Epoch: {epoch}"
            
            QMessageBox.information(
                self, 
                "Load Config And Model", 
                f"Configuration and model loaded from checkpoint:\n{file_path}\n\n"
                f"Model type: {model_type.upper()}\n"
                f"Input dim: {input_dim}\n"
                f"Sequence length: {sequence_length}{metrics_message}\n\n"
                f"Note: Feature set, standardization, and normalization controls are now frozen.{missing_elements_message}"
            )
            
        except Exception as e:
            prError(f"Failed to load config and model from checkpoint: {e}")
            QMessageBox.critical(self, "Load Config And Model Error", str(e))
            import traceback
            traceback.print_exc()
    
    def create_dataset(self):
        """Create dataset with current configuration"""
        config = self.get_config()
        self.validate_config()  # This will raise an exception if invalid
        # Find the main window ancestor and emit the create_dataset_signal
        main_window = self.window()
        if hasattr(main_window, 'create_dataset_signal'):
            main_window.create_dataset_signal.emit(config)
        else:
            raise RuntimeError("Main window does not have create_dataset_signal")


class ShelfyVisualizerMainWindow(QMainWindow):
    """Main window for the Shelfy dataset visualizer"""
    
    create_dataset_signal = pyqtSignal(dict)
    
    def __init__(self, raw_data_path=None):
        super().__init__()
        self.dataset = None
        self.dataset_worker = None
        self.raw_data_path = raw_data_path
        
        self.setup_ui()
        self.setup_connections()
        
        # Set window properties
        self.setWindowTitle("Shelfy Dataset Visualizer")
        self.setGeometry(100, 100, 1800, 1000)
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout()
        
        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Parameter controls
        self.parameter_controls = ParameterControlsWidget()
        self.parameter_controls.setMaximumWidth(450)
        splitter.addWidget(self.parameter_controls)
        
        # Right panel - Data visualization
        self.data_visualization = DataVisualizationWidget(self.raw_data_path)
        splitter.addWidget(self.data_visualization)
        
        # Set splitter proportions
        splitter.setSizes([450, 750])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def setup_connections(self):
        """Setup signal connections"""
        self.create_dataset_signal.connect(self.create_dataset)
    
    def create_dataset(self, config):
        """Create dataset with given configuration"""
        self.status_bar.showMessage("Creating dataset...")
        
        # Create worker thread
        self.dataset_worker = DatasetWorker(config)
        self.dataset_worker.dataset_created.connect(self.on_dataset_created)
        self.dataset_worker.error_occurred.connect(self.on_dataset_error)
        self.dataset_worker.start()
    
    def on_dataset_created(self, dataset, message):
        """Handle successful dataset creation"""
        self.dataset = dataset
        self.data_visualization.set_dataset(dataset)
        
        # Check if model is loaded and verify compatibility
        if self.data_visualization.current_model is not None:
            # Check if dataset input dimensions match model
            dataset_input_dim = len(dataset.data_columns_in_dataset)
            if isinstance(self.data_visualization.current_model, STG_NF):
                model_input_dim = -1
            elif isinstance(self.data_visualization.current_model, STGCN):
                # STGCN models don't use input_dim in the same way
                model_input_dim = -1
            elif isinstance(self.data_visualization.current_model, SkateFormer):
                # SkateFormer models don't use input_dim in the same way
                model_input_dim = -1
            else:
                model_input_dim = self.data_visualization.current_model.input_dim
            
            if dataset_input_dim != model_input_dim and model_input_dim != -1:
                QMessageBox.warning(
                    self,
                    "Model-Dataset Mismatch",
                    f"Dataset input dimension ({dataset_input_dim}) does not match "
                    f"model input dimension ({model_input_dim}).\n\n"
                    f"Please load a compatible dataset or reload the model with matching config."
                )
            else:
                # Model is compatible, refresh display to show predictions
                self.data_visualization.display_current_item()
        
        self.status_bar.showMessage(f"Dataset created: {len(dataset)} items")
        
        QMessageBox.information(self, "Success", message)
    
    def on_dataset_error(self, error_message):
        """Handle dataset creation error"""
        self.status_bar.showMessage("Dataset creation failed")
        QMessageBox.critical(self, "Error", error_message)


def main(args):
    """Main function to run the application"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Shelfy Dataset Visualizer")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = ShelfyVisualizerMainWindow(args.raw_data_path)
    window.show()
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset Visualizer")
    parser.add_argument("--raw_data_path", "-d", type=str, default="/media/raphael/hucedisk3/HUI360-Video/", help="Path to the raw data")
    args = parser.parse_args()
    
    main(args)
