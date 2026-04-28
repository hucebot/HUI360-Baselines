from torch.utils.data import Dataset
import torch
import pickle
import os
import glob

# Keypoint names for augmentation (left/right swapping)
VITPOSE_KEYPOINTS_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", 
                           "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
                           "left_wrist", "right_wrist", "left_hip", "right_hip", 
                           "left_knee", "right_knee", "left_ankle", "right_ankle"]

# Mapping for left/right keypoint swapping during horizontal flip
LEFT_RIGHT_SWAP_PAIRS = [
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]

# COCO17 joint indices for [T, 17, C]: (left_idx, right_idx) for horizontal flip
# Order: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, ...
COCO17_SWAP_JOINT_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

# Non-stochastic parameters that define dataset preprocessing (used for matching)
# These are the parameters that, if identical, mean a preloaded dataset can be reused
NON_STOCHASTIC_CONFIG_KEYS = [
    "include_recordings",
    "include_tracks",
    "include_columns",
    "positive_cutoff",
    "interaction_cutoff",
    "cutoffs_filtering",
    "fixed_input_length",
    "input_length_in_frames",
    "min_length_in_frames",
    "max_length_in_frames",
    "subsample_frames",
    "min_keypoints_filter",
    "additional_filtering_dict",
    "normalize_in_image",
    "normalize_keypoints_in_box",
    "normalize_keypoints_in_track",
    "do_recenter_interaction_zone",
    "standardize_data",
    "fix_index_per_track",
    "fix_index_per_track_list",
    "force_positive_samples",
    "force_aligment_with_biggest_mask_size",
    "center_on_onset",
    "do_recentering",
    "do_fix_keypoints_outside_box",
    "inputs_per_track_stride",
    "hf_dataset_revision",
    "format_by_channel",
    "perspective_reprojection",
    "remove_joints",
]


def build_preload_config(preloaded_config, dataset_type="train"):
    """
    Build a config dict with non-stochastic parameters for matching preloaded datasets.
    
    Args:
        preloaded_config: Full config dictionary from config file
        dataset_type: "train" or "val" or "downstream_train" or "downstream_val" to get the right parameters
        
    Returns:
        Dict with non-stochastic config parameters
    """
    suffix = "_" + dataset_type #train" if "train" in dataset_type else "_val"
    config = {}
    
    for key in NON_STOCHASTIC_CONFIG_KEYS:
        # Some keys have train/val suffix, some don't
        if key + suffix in preloaded_config:
            config[key] = preloaded_config[key + suffix]
        elif key in preloaded_config:
            config[key] = preloaded_config[key]
        # Keys that might not exist in all configs
    
    # config["dataset_type"] = dataset_type # no need to save it, we dont care about the dataset type for future loading
    return config


def configs_match(config1, config2):
    """
    Check if two preload configs match (all non-stochastic parameters are identical).
    
    Args:
        config1: First config dict
        config2: Second config dict
        
    Returns:
        True if configs match, False otherwise
    """
    if config1.keys() != config2.keys():
        # print(f"Keys mismatch: {config1.keys()} != {config2.keys()}")
        return False
    
    for key in config1:
        val1 = config1[key]
        val2 = config2[key]
        
        # Handle list comparison
        if isinstance(val1, list) and isinstance(val2, list):
            if val1 != val2:
                return False
        # Handle dict comparison
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if val1 != val2:
                return False
        # Handle other types
        elif val1 != val2:
            return False
    
    return True


def find_matching_preloaded_dataset(save_dir, config):
    """
    Find an existing preloaded dataset that matches the given config.
    
    Args:
        save_dir: Directory where preloaded datasets are stored
        config: Config dict with non-stochastic parameters
        
    Returns:
        Path to matching preloaded dataset, or None if not found
    """
    if not os.path.exists(save_dir):
        return None
    
    # Find all config files for the given dataset type
    config_pattern = os.path.join(save_dir, f"preloaded_*_config.pkl")
    config_files = glob.glob(config_pattern)
    
    for config_file in config_files:
        try:
            with open(config_file, "rb") as f:
                existing_config = pickle.load(f)
            
            if configs_match(config, existing_config):
                # Found a match - return the corresponding data file
                data_file = config_file.replace("_config.pkl", ".pkl")
                if os.path.exists(data_file):
                    return data_file
        except Exception as e:
            # Skip corrupted or unreadable config files
            print(f"Warning: Could not read config file {config_file}: {e}")
            continue
    
    return None


class HUIInteract360Light(Dataset):
    """
    Light dataset class for preloaded/preprocessed data.
    
    Loads preloaded pkl files created by preload_dataset() function.
    When augmentation is enabled, random_flip and random_jitter are applied
    directly on the normalized tensors.
    """
    
    def __init__(self, dataset_path, 
                 random_flip_horizontal=False,
                 random_jitter_position=(0.0, 0.0)):
        """
        Args:
            dataset_path: Path to the preprocessed pkl file (created by preload_dataset)
            random_flip_horizontal: Whether to apply random horizontal flip augmentation
            random_jitter_position: Tuple (max_jitter_x, max_jitter_y) for random jitter augmentation
        """
        super().__init__()
        
        light_dataset_pickle = pickle.load(open(dataset_path, "rb"))
        
        self.inputs = light_dataset_pickle["inputs"]
        self.labels = light_dataset_pickle["labels"]
        self.metadata = light_dataset_pickle["metadata"]
        
        # ### Sample 120 positives and 12 0 negatives to be used for downstream eval of models (100/20) ###
        # import random
        # all_positives = [i for i, label in enumerate(self.labels) if label == 1]
        # all_negatives = [i for i, label in enumerate(self.labels) if label == 0]
        # positives = random.sample(all_positives, 100)
        # negatives = random.sample(all_negatives, 100)
        
        # print(len(self.inputs), len(self.labels), len(self.metadata))
        # exit()
        # Augmentation settings
        self.random_flip_horizontal = random_flip_horizontal
        self.random_jitter_position = random_jitter_position
        
        # Load preload metadata
        preload_meta = light_dataset_pickle["preload_metadata"]
        self.data_columns_in_dataset = preload_meta["data_columns_in_dataset"]
        self.input_length_in_frames = preload_meta["input_length_in_frames"]
        self.subsample_frames = preload_meta["subsample_frames"]
        self.standardize_data = preload_meta["standardize_data"]
        self.normalize_keypoints_in_box = preload_meta["normalize_keypoints_in_box"]
        self.normalize_keypoints_in_track = preload_meta["normalize_keypoints_in_track"]
        self.normalize_in_image = preload_meta["normalize_in_image"]
        
        # Build column index mappings for efficient augmentation
        self._build_column_indices()
        
        # Dataset statistics
        self.total_positives_tracks = preload_meta["total_positives_tracks"]
        self.total_negatives_tracks = preload_meta["total_negatives_tracks"]
        self.total_possible_positives_segments = preload_meta["total_possible_positives_segments"]
        self.total_possible_negatives_segments = preload_meta["total_possible_negatives_segments"]
        self.total_used_positive_segments = preload_meta["total_used_positive_segments"]
        self.total_used_negative_segments = preload_meta["total_used_negative_segments"]
        
        self.total_samples = len(self.labels)
    
    def _build_column_indices(self):
        """Build column index mappings for efficient tensor augmentation."""
        self.col_to_idx = {col: idx for idx, col in enumerate(self.data_columns_in_dataset)}
        
        # Find indices for x and y columns (for flip and jitter)
        self.x_col_indices = []
        self.y_col_indices = []
        
        # Box columns
        self.xmin_idx = self.col_to_idx.get('xmin')
        self.xmax_idx = self.col_to_idx.get('xmax')
        self.ymin_idx = self.col_to_idx.get('ymin')
        self.ymax_idx = self.col_to_idx.get('ymax')
        
        if self.xmin_idx is not None:
            self.x_col_indices.append(self.xmin_idx)
        if self.xmax_idx is not None:
            self.x_col_indices.append(self.xmax_idx)
        if self.ymin_idx is not None:
            self.y_col_indices.append(self.ymin_idx)
        if self.ymax_idx is not None:
            self.y_col_indices.append(self.ymax_idx)
        
        # Keypoint columns
        self.keypoint_x_indices = {}  # keypoint_name -> column index
        self.keypoint_y_indices = {}
        
        for kp_name in VITPOSE_KEYPOINTS_NAMES:
            x_col = f"vitpose_{kp_name}_x"
            y_col = f"vitpose_{kp_name}_y"
            
            if x_col in self.col_to_idx:
                self.keypoint_x_indices[kp_name] = self.col_to_idx[x_col]
                self.x_col_indices.append(self.col_to_idx[x_col])
            if y_col in self.col_to_idx:
                self.keypoint_y_indices[kp_name] = self.col_to_idx[y_col]
                self.y_col_indices.append(self.col_to_idx[y_col])
        
        # Build swap pairs for horizontal flip (left <-> right)
        self.swap_pairs_x = []  # list of (left_idx, right_idx) for x columns
        self.swap_pairs_y = []  # list of (left_idx, right_idx) for y columns
        self.swap_pairs_score = []  # list of (left_idx, right_idx) for score columns
        
        for left_name, right_name in LEFT_RIGHT_SWAP_PAIRS:
            if left_name in self.keypoint_x_indices and right_name in self.keypoint_x_indices:
                self.swap_pairs_x.append((self.keypoint_x_indices[left_name], self.keypoint_x_indices[right_name]))
            if left_name in self.keypoint_y_indices and right_name in self.keypoint_y_indices:
                self.swap_pairs_y.append((self.keypoint_y_indices[left_name], self.keypoint_y_indices[right_name]))
            
            # Also swap scores
            left_score_col = f"vitpose_{left_name}_score"
            right_score_col = f"vitpose_{right_name}_score"
            if left_score_col in self.col_to_idx and right_score_col in self.col_to_idx:
                self.swap_pairs_score.append((self.col_to_idx[left_score_col], self.col_to_idx[right_score_col]))
    
    def _flip_tensor_by_channel_coco(self, input_tensor, image_width):
        """ Flip the input tensor when in format [T, 17, C] (COCO format)

        Joints are = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

        Args:
            input_tensor (torch.Tensor): input tensor of shape [T, 17, C]
            image_width: Image width
            
        Returns:
            torch.Tensor: flipped tensor of shape [T, 17, C]
        """
        flipped = input_tensor.clone()
        T, V, C = flipped.shape

        assert(not self.normalize_keypoints_in_box), "normalize_keypoints_in_box must be False when using format_by_channel in Light dataset"
        assert(self.normalize_keypoints_in_track == "none"), "normalize_keypoints_in_track must be 'none' when using format_by_channel in Light dataset"
        
        # Flip x coordinate (channel 0) for all joints
        if C > 0:
            if self.standardize_data == "all":
                flipped[..., 0] = -input_tensor[..., 0]
            elif self.normalize_in_image:
                flipped[..., 0] = 1.0 - input_tensor[..., 0]
            else:
                flipped[..., 0] = image_width - input_tensor[..., 0]

        # Swap left/right joint pairs (entire joint rows)
        for left_idx, right_idx in COCO17_SWAP_JOINT_PAIRS:
            if left_idx < V and right_idx < V:
                tmp = flipped[:, left_idx, :].clone()
                flipped[:, left_idx, :] = flipped[:, right_idx, :]
                flipped[:, right_idx, :] = tmp

        return flipped
    
    def _flip_tensor(self, input_tensor, image_width):
        """
        Apply horizontal flip to the input tensor.
        
        For standardized data, flipping negates x values (since mean≈0.5 for normalized data).
        Also swaps left/right keypoints and xmin/xmax.
        
        Args:
            input_tensor: Tensor of shape (T, D) where T is time and D is features
            image_width: Image width
            
        Returns:
            Flipped tensor of shape (T, D)
        """
        flipped = input_tensor.clone()
        
        assert(not self.normalize_keypoints_in_box), "normalize_keypoints_in_box must be False when using format_by_channel in Light dataset"
        assert(self.normalize_keypoints_in_track == "none"), "normalize_keypoints_in_track must be 'none' when using format_by_channel in Light dataset"
        
        # Flip all x coordinates
        if self.standardize_data == "all":
            # For already standardized data, flipping negates x values
            # (since mean≈0.5 for normalized [0,1] data, (1-2*0.5)/std ≈ 0)
            for idx in self.x_col_indices:
                flipped[:, idx] = -input_tensor[:, idx]
        elif self.normalize_in_image:
            # For non-standardized (just normalized to [0,1]) data
            for idx in self.x_col_indices:
                flipped[:, idx] = 1.0 - input_tensor[:, idx]
        else:
            for idx in self.x_col_indices:
                flipped[:, idx] = image_width - input_tensor[:, idx]
                
        # Handle box coordinates: xmin <-> xmax swap
        if self.xmin_idx is not None and self.xmax_idx is not None:
            old_xmin = flipped[:, self.xmin_idx].clone()
            old_xmax = flipped[:, self.xmax_idx].clone()
            flipped[:, self.xmin_idx] = old_xmax
            flipped[:, self.xmax_idx] = old_xmin
        
        # Swap left/right keypoints
        for left_idx, right_idx in self.swap_pairs_x:
            old_left = flipped[:, left_idx].clone()
            flipped[:, left_idx] = flipped[:, right_idx]
            flipped[:, right_idx] = old_left
            
        for left_idx, right_idx in self.swap_pairs_y:
            old_left = flipped[:, left_idx].clone()
            flipped[:, left_idx] = flipped[:, right_idx]
            flipped[:, right_idx] = old_left
            
        for left_idx, right_idx in self.swap_pairs_score:
            old_left = flipped[:, left_idx].clone()
            flipped[:, left_idx] = flipped[:, right_idx]
            flipped[:, right_idx] = old_left
        
        return flipped
    
    def _jitter_tensor(self, input_tensor, jitter_x, jitter_y):
        """
        Apply position jitter to the input tensor.
        
        Args:
            input_tensor: Tensor of shape (T, D)
            jitter_x: Jitter amount for x coordinates (in normalized [0,1] space)
            jitter_y: Jitter amount for y coordinates (in normalized [0,1] space)
            
        Returns:
            Jittered tensor of shape (T, D)
        """
        jittered = input_tensor.clone()
        
        if self.standardize_data == "all":
            # Convert jitter to standardized space using approximate std values
            jitter_x_std = jitter_x / 0.18  # approximate std for x coords
            jitter_y_std = jitter_y / 0.06  # approximate std for y coords
            
            for idx in self.x_col_indices:
                jittered[:, idx] = input_tensor[:, idx] + jitter_x_std
            for idx in self.y_col_indices:
                jittered[:, idx] = input_tensor[:, idx] + jitter_y_std
        else:
            # For non-standardized data, add jitter directly
            for idx in self.x_col_indices:
                jittered[:, idx] = input_tensor[:, idx] + jitter_x
            for idx in self.y_col_indices:
                jittered[:, idx] = input_tensor[:, idx] + jitter_y
        
        return jittered
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = self.inputs[idx]
        label = self.labels[idx]
        metadata = self.metadata[idx]

        metadata["orig_dataset"] = "HUI360"
        metadata["file_path"] = "UnknownFilePath"
        
        # Convert to tensor if numpy array
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        else:
            input_tensor = input_tensor.clone()
        
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        
        # Apply augmentations if enabled
        if self.random_flip_horizontal and torch.rand(1).item() < 0.5:
            if input_tensor.ndim == 3:
                input_tensor = self._flip_tensor_by_channel_coco(input_tensor, metadata["image_size"][0])
            else:
                input_tensor = self._flip_tensor(input_tensor, metadata["image_size"][0])
            if isinstance(metadata, dict):
                metadata = metadata.copy()
                metadata["flipped"] = True
        
        if self.random_jitter_position[0] > 0.0 or self.random_jitter_position[1] > 0.0:
            jitter_x = (torch.rand(1).item() - 0.5) * 2 * self.random_jitter_position[0]
            jitter_y = (torch.rand(1).item() - 0.5) * 2 * self.random_jitter_position[1]
            input_tensor = self._jitter_tensor(input_tensor, jitter_x, jitter_y)
            if isinstance(metadata, dict):
                metadata = metadata.copy()
                metadata["jitter_x_pct"] = jitter_x
                metadata["jitter_y_pct"] = jitter_y
        
        # Return format compatible with the main dataset (5-tuple)
        return input_tensor, label, metadata, [], []


def preload_dataset(dataset, config, save_dir):
    """
    Preload a HUIInteract360 dataset and save it for faster loading.
    Also saves a separate config file for fast matching on future runs.
    
    Args:
        dataset: HUIInteract360 dataset instance (with stochastic augmentations disabled)
        config: Config dict with non-stochastic parameters (for matching)
        save_dir: Directory to save the preloaded data
        
    Returns:
        Path to the saved pkl file
    """
    import time
    from tqdm import tqdm
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"preloaded_{timestamp}_cols{len(dataset.data_columns_in_dataset)}_len{len(dataset)}_seq{dataset.input_length_in_frames}.pkl"
    filepath = os.path.join(save_dir, filename)
    config_filepath = filepath.replace(".pkl", "_config.pkl")
    
    inputs = []
    labels = []
    metadata_list = []
    
    print(f"Preloading dataset ({len(dataset)} samples)...")
    for idx in tqdm(range(len(dataset)), desc=f"Preloading"):
        input_tensor, label, metadata, _, _ = dataset[idx]
        inputs.append(input_tensor.numpy() if isinstance(input_tensor, torch.Tensor) else input_tensor)
        labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        metadata_list.append(metadata)
    
    # Build preload metadata used to match preloaded datasets
    preload_metadata = {
        "data_columns_in_dataset": dataset.data_columns_in_dataset,
        "input_length_in_frames": dataset.input_length_in_frames,
        "subsample_frames": dataset.subsample_frames,
        "standardize_data": dataset.standardize_data,
        "normalize_keypoints_in_box": dataset.normalize_keypoints_in_box,
        "normalize_keypoints_in_track": dataset.normalize_keypoints_in_track,
        "normalize_in_image": dataset.normalize_in_image,
        "total_positives_tracks": dataset.total_positives_tracks,
        "total_negatives_tracks": dataset.total_negatives_tracks,
        "total_possible_positives_segments": dataset.total_possible_positives_segments,
        "total_possible_negatives_segments": dataset.total_possible_negatives_segments,
        "total_used_positive_segments": getattr(dataset, 'total_used_positive_segments', None),
        "total_used_negative_segments": getattr(dataset, 'total_used_negative_segments', None),
        "timestamp": timestamp,
        # "dataset_type": dataset_type, # no need to save it, we dont care about the dataset type here
    }
    
    data_to_save = {
        "inputs": inputs,
        "labels": labels,
        "metadata": metadata_list,
        "preload_metadata": preload_metadata,
    }
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save data file
    with open(filepath, "wb") as f:
        pickle.dump(data_to_save, f)
    
    # Save config file separately for fast matching
    with open(config_filepath, "wb") as f:
        pickle.dump(config, f)
    
    print(f"Saved preloaded dataset to: {filepath}")
    print(f"Saved preload config to: {config_filepath}")
    return filepath
