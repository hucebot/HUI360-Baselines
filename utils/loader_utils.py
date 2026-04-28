## Plotting function for debug...
import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join("..", here))

from utils.print_utils import prInfo, prSuccess, prWarning, prError, prDebug
from datasets.HUIDataset import HUIInteract360
from datasets.HUIDatasetLight import (
    HUIInteract360Light, 
    preload_dataset, 
    build_preload_config, 
    find_matching_preloaded_dataset
)

from tools.create_config_files import (
    HUI_TRAININGSET, HUI_TESTINGSET, SSUP_TRAININGSET, SSUP_TESTINGSET,
    FEATURES_SET_D1, FEATURES_SET_D2, FEATURES_SET_D3, FEATURES_SET_D4, FEATURES_SET_D5, FEATURES_SET_D6, FEATURES_SET_D7
)

FEATURE_SET_NAME_TO_LIST = {
    "features_set_d1": FEATURES_SET_D1,
    "features_set_d2": FEATURES_SET_D2,
    "features_set_d3": FEATURES_SET_D3,
    "features_set_d4": FEATURES_SET_D4,
    "features_set_d5": FEATURES_SET_D5,
    "features_set_d6": FEATURES_SET_D6,
    "features_set_d7": FEATURES_SET_D7
}


def load_hui_dataset(args: dict, config: dict, split: str="train", num_workers: int=0) -> HUIInteract360 | HUIInteract360Light:
    if config[f"{split}_tracks_filename"] != "all":
        tracks = [line.strip() for line in open(os.path.join(here, "..", "datasets", "tracks_saved_identifiers", config[f"{split}_tracks_filename"]))]
    else:
        tracks = "all"
    
    # Store augmentation settings for potential preloading
    random_flip = config[f"random_flip_horizontal_{split}"]
    random_jitter = (config[f"random_jitter_position_{split}"][0], config[f"random_jitter_position_{split}"][1])
    
    # Directory for preloaded datasets
    temp_data_dir = os.path.join(here, "..", "datasets", "temp_data")
        
    if args.preload_data or args.preload_only:
        os.makedirs(temp_data_dir, exist_ok=True)
        
        # Build config for matching existing preloaded datasets
        preload_config = build_preload_config(config, dataset_type=split)
        # Add tracks to config (it's not in config)
        preload_config["include_tracks"] = tracks
        
        # Check for existing matching preloaded training dataset
        preloaded_path = find_matching_preloaded_dataset(temp_data_dir, preload_config)
        
        # #### debug drop to json
        # print(preload_config)
        # from utils.other_utils import write_dic_to_json_file
        # import time
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        # write_dic_to_json_file(preload_config, os.path.join(here, "..", f"preload_config_{split}_{config['experiment_name']}_{timestamp}.json"))
        # ####################
        
        if preloaded_path is not None:
            prSuccess(f"Found matching preloaded {split} dataset: {preloaded_path}")
        else:
            prInfo(f"No matching preloaded {split} dataset found, creating new one...")
            prInfo(f"Creating {split} dataset for preloading (stochastic augmentations disabled)...")
            dataset_for_preload = HUIInteract360(
                include_recordings=config[f"include_recordings_{split}"], 
                include_tracks=tracks, 
                include_columns=config["include_columns"],
                positive_cutoff=config[f"positive_cutoff_{split}"],
                interaction_cutoff=config[f"interaction_cutoff_{split}"],
                cutoffs_filtering=config[f"cutoffs_filtering"],
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
                normalize_keypoints_in_track=config["normalize_keypoints_in_track"],
                do_recenter_interaction_zone=config["do_recenter_interaction_zone"],
                random_flip_horizontal=False,  # Disabled for preloading
                random_jitter_position=(0.0, 0.0),  # Disabled for preloading
                standardize_data=config["standardize_data"],
                fix_index_per_track=config[f"fix_index_per_track_{split}"],
                fix_index_per_track_list=config[f"fix_index_per_track_list_{split}"],
                force_positive_samples=config["force_positive_samples"],
                force_aligment_with_biggest_mask_size=config[f"force_aligment_with_biggest_mask_size_{split}"],
                verbose=args.verbose,
                center_on_onset=config[f"center_on_onset_{split}"],
                do_recentering=config[f"do_recentering_{split}"],
                do_fix_keypoints_outside_box=config[f"do_fix_keypoints_outside_box_{split}"],
                inputs_per_track_stride=config[f"inputs_per_track_stride_{split}"],
                hf_dataset_revision=config["hf_dataset_revision"],
                allow_download=True,
                hf_local_dir=args.hf_local_dir,
                num_workers=num_workers,
                format_by_channel=config["format_by_channel"],
                offline_mode=args.offline_mode,
                perspective_reprojection=config.get("perspective_reprojection", None),
                remove_joints=config.get("remove_joints", None),
            )
            
            # Preload and save the training dataset
            preloaded_path = preload_dataset(
                dataset_for_preload, 
                preload_config, 
                temp_data_dir
            )
            
            # Clean up the original dataset to free memory
            del dataset_for_preload
        
        # Load the preloaded dataset with augmentation support
        dataset = HUIInteract360Light(
            preloaded_path,
            random_flip_horizontal=random_flip,
            random_jitter_position=random_jitter
        )
    else:
        dataset = HUIInteract360(
            include_recordings=config[f"include_recordings_{split}"], 
            include_tracks=tracks, 
            include_columns=config["include_columns"],
            positive_cutoff=config[f"positive_cutoff_{split}"],
            interaction_cutoff=config[f"interaction_cutoff_{split}"],
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
            normalize_keypoints_in_track=config["normalize_keypoints_in_track"],
            do_recenter_interaction_zone=config["do_recenter_interaction_zone"],
            random_flip_horizontal=config[f"random_flip_horizontal_{split}"],
            random_jitter_position=(config[f"random_jitter_position_{split}"][0], config[f"random_jitter_position_{split}"][1]),
            standardize_data=config["standardize_data"],
            fix_index_per_track=config[f"fix_index_per_track_{split}"],
            fix_index_per_track_list=config[f"fix_index_per_track_list_{split}"],
            force_positive_samples=config["force_positive_samples"],
            force_aligment_with_biggest_mask_size=config[f"force_aligment_with_biggest_mask_size_{split}"],
            verbose=args.verbose,
            center_on_onset=config[f"center_on_onset_{split}"],
            do_recentering=config[f"do_recentering_{split}"],
            do_fix_keypoints_outside_box=config[f"do_fix_keypoints_outside_box_{split}"],
            inputs_per_track_stride=config[f"inputs_per_track_stride_{split}"],
            hf_dataset_revision=config["hf_dataset_revision"],
            format_by_channel=config["format_by_channel"],
            allow_download=True,
            hf_local_dir=args.hf_local_dir,
            num_workers=num_workers,
            offline_mode=args.offline_mode,
            perspective_reprojection=config.get("perspective_reprojection", None),
            remove_joints=config.get("remove_joints", None),
        )
    
    return dataset
