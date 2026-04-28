import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, ".."))

from math import e
from huggingface_hub import snapshot_download, hf_hub_download
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import numpy as np
import pandas as pd
from typing import Optional, Union
import torch.multiprocessing as mp
from torchvision import transforms as T

from utils.print_utils import *
from utils.rle_tools import *
from utils.data_utils import *
from utils.other_utils import read_json_to_dic

from datasets.hui_norm_values import HUI_NORMALIZATION_VALUES
from datasets.HUIDatasetUtils import *

from PIL import Image

from datasets.HUIDatasetUtils import input_tensor_to_format_by_channel

# Ensure reproducibility with seeds (including for the random augmentations)
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_image_path(raw_dataset_path: Union[str, None], recording:str, episode:str, image_file: str) -> str:
    if raw_dataset_path is None:
        raw_dataset_path = "/media/raphael/hucedisk/ShelfyInteractFull/data_extractor/"
    images_dir = "images_360"
    return os.path.join(raw_dataset_path, recording, "episodes", episode, images_dir, image_file)
    

DEFAULT_IMAGE_TRANSFORMS = [
    T.Resize((256, 256)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

DEFAULT_MASK_TRANSFORMS = [
    T.Resize((256, 256)),
]


class HUIInteract360(Dataset):
    
    def __init__(self,
                 include_recordings: Union[list, str] = "all",
                 include_tracks: Union[list, str] = "all",
                 include_columns: Union[list, str] = "all",
                 positive_cutoff: int = 60, # in frames
                 interaction_cutoff: int = 5, # in frames
                 cutoffs_filtering: bool = True,
                 fixed_input_length: bool = True,
                 input_length_in_frames: Optional[int] = 10, # if fixed_input_length is True then it should be None
                 min_length_in_frames: Optional[int] = None,
                 max_length_in_frames: Optional[int] = None,
                 subsample_frames: Optional[int] = 1,
                 min_keypoints_filter: int = 9,
                 additional_filtering_dict: dict = {"mask_size": {"min": 1000, "max": 1e7}},
                 raw_dataset_path: str = None,
                 image_transforms: list = DEFAULT_IMAGE_TRANSFORMS,
                 mask_transforms: list = DEFAULT_MASK_TRANSFORMS,
                 return_images: bool = False,
                 return_masks: bool = False,
                 normalize_in_image: bool = True,
                 normalize_keypoints_in_box: bool = False,
                 normalize_keypoints_in_track: str = "norm_xy",
                 do_recenter_interaction_zone: bool = False,
                 random_flip_horizontal: bool = False,
                 random_jitter_position: tuple = (0.0, 0.0), # (maximum jitter_x_pct, maximum jitter_y_pct)
                 standardize_data: str = "all",
                 metadata_columns: list = METADATA_COLUMNS,
                 fix_index_per_track: bool = False,
                 fix_index_per_track_list: list = None,
                 force_positive_samples: bool = False,
                 force_aligment_with_biggest_mask_size: bool = False,
                 verbose: bool = True,
                 center_on_onset: bool = False,
                 do_recentering: bool = False,
                 do_fix_keypoints_outside_box: bool = True, # should always be done, but keep the option to test with and without it
                 inputs_per_track_stride: int = -1, # if -1, then a single input is used for each track, if > 0, then only every inputs_per_track_stride inputs are used
                 allow_download: bool = True,
                 hf_dataset_revision: str = "3c8a342548534b6b92d32b0099e266962facdf45",
                 hf_local_dir: str = "default", #os.path.join(here, "..", "datasets", "hf_data"),
                 num_workers: int = 0,
                 format_by_channel: bool = False, # format the input tensor to be a COCO-like skeleton format (B,T,V,C) instead of (B,T,D), requires all ViTPose keypoints to be present in the data columns, and this is done at __getitem__ time
                 offline_mode: bool = False,
                 perspective_reprojection: dict = None,
                 remove_joints: list = None,
                 ):
        
        if force_aligment_with_biggest_mask_size and center_on_onset:
            raise ValueError("force_aligment_with_biggest_mask_size and center_on_onset cannot be True at the same time")
        
        if fixed_input_length:
            assert(min_length_in_frames is None), "If fixed_input_length is True, min_length_in_frames must be None"
            assert(max_length_in_frames is None), "If fixed_input_length is True, max_length_in_frames must be None"
            assert(input_length_in_frames%subsample_frames == 0), "input_length_in_frames must be divisible by subsample_frames"
            assert(input_length_in_frames >= subsample_frames), "input_length_in_frames must be greater than or equal to subsample_frames"
        else:
            assert(input_length_in_frames is None), "If fixed_input_length is False, input_length_in_frames must be None"
            assert(min_length_in_frames is not None), "If fixed_input_length is False, min_length_in_frames must be not None"
            assert(max_length_in_frames is not None), "If fixed_input_length is False, max_length_in_frames must be not None"
            assert(min_length_in_frames <= max_length_in_frames), "min_length_in_frames must be less than or equal to max_length_in_frames"
            assert(min_length_in_frames >= subsample_frames), "min_length_in_frames must be greater than or equal to subsample_frames"
        
        assert(standardize_data in ["all", "none", "mask_only"]), "standardize_data must be either 'all' or 'none' or 'mask_only'"
        
        # if return_images or return_masks:
        #     raise NotImplementedError("Not implemented to return images or masks")
        
        if include_columns != "all":
            assert("recording" in include_columns), "recording column must be in include_columns"
            assert("unique_track_identifier" in include_columns), "unique_track_identifier column must be in include_columns"
            assert("image_index" in include_columns), "image_index column must be in include_columns"
        
        if normalize_keypoints_in_box:
            assert(normalize_keypoints_in_track == "none"), "normalize_keypoints_in_box and normalize_keypoints_in_track cannot be enabled at the same time"
            assert(normalize_in_image), "normalize_keypoints_in_box requires normalize_in_image to be True"
        
        if normalize_keypoints_in_track != "none":
            assert(normalize_keypoints_in_track in ["norm_xy", "norm_x", "norm_y"]), "normalize_keypoints_in_track must be either 'norm_xy', 'norm_x' or 'norm_y'"
            assert(normalize_in_image), "normalize_keypoints_in_track requires normalize_in_image to be True"
            assert(not normalize_keypoints_in_box), "normalize_keypoints_in_track requires normalize_keypoints_in_box to be False"
            
        if fix_index_per_track_list is not None:
            assert(fix_index_per_track), "fix_index_per_track_list requires fix_index_per_track to be True"
        
        # Set variables
        self.include_recordings = include_recordings
        self.include_tracks = include_tracks
        self.include_columns = include_columns
        self.positive_cutoff = positive_cutoff
        self.interaction_cutoff = interaction_cutoff
        self.cutoffs_filtering = cutoffs_filtering
        self.fixed_input_length = fixed_input_length
        self.input_length_in_frames = input_length_in_frames
        self.min_length_in_frames = min_length_in_frames
        self.max_length_in_frames = max_length_in_frames
        self.subsample_frames = subsample_frames
        self.min_keypoints_filter = min_keypoints_filter
        self.additional_filtering_dict = additional_filtering_dict
        self.force_positive_samples = force_positive_samples
        self.force_aligment_with_biggest_mask_size = force_aligment_with_biggest_mask_size
        self.raw_dataset_path = raw_dataset_path
        self.return_images = return_images
        self.return_masks = return_masks
        self.image_transforms = T.Compose(image_transforms)
        self.mask_transforms = T.Compose(mask_transforms)
        self.normalize_in_image = normalize_in_image
        self.normalize_keypoints_in_track = normalize_keypoints_in_track
        self.normalize_keypoints_in_box = normalize_keypoints_in_box
        self.do_recenter_interaction_zone = do_recenter_interaction_zone
        self.random_flip_horizontal = random_flip_horizontal
        self.random_jitter_position = random_jitter_position
        self.standardize_data = standardize_data
        self.metadata_columns = metadata_columns + ["mask_size_meta", "xmin_meta", "xmax_meta", "ymin_meta", "ymax_meta"] + VITPOSE_KEYPOINTS_SCORES_COLUMNS # append bounding box columns used for normalization (even in the absence of bounding box in the actual data input), created at loading time
        self.verbose = verbose
        self.fix_index_per_track = fix_index_per_track
        self.center_on_onset = center_on_onset
        self.do_recentering = do_recentering
        self.do_fix_keypoints_outside_box = do_fix_keypoints_outside_box
        self.inputs_per_track_stride = inputs_per_track_stride
        self.num_workers = num_workers
        self.offline_mode = offline_mode
        self.format_by_channel = format_by_channel
        self.remove_joints = remove_joints
        
        if perspective_reprojection is not None and perspective_reprojection.get("do_perspective_reprojection", False):
            self.do_perspective_reprojection = True
            self.perspective_hcenter = perspective_reprojection.get("hcenter", 0.0)
            self.perspective_vcenter = perspective_reprojection.get("vcenter", 0.0)
            self.perspective_hfov = perspective_reprojection.get("hfov", 94.0)
            self.perspective_vfov = perspective_reprojection.get("vfov", 68.0)
            self.perspective_target_width = perspective_reprojection.get("target_width", 1280)
            self.perspective_target_height = int(round(
                self.perspective_target_width * np.tan(np.radians(self.perspective_vfov / 2))
                / np.tan(np.radians(self.perspective_hfov / 2))
            ))
            self.log(f"Perspective reprojection enabled: hcenter={self.perspective_hcenter}, vcenter={self.perspective_vcenter}, "
                     f"hfov={self.perspective_hfov}, vfov={self.perspective_vfov}, "
                     f"target_size=({self.perspective_target_width}, {self.perspective_target_height})", "info")
        else:
            self.do_perspective_reprojection = False
        
        if self.format_by_channel:
            vitpose_cols = [col for col in self.include_columns if col.startswith("vitpose")]
            assert(len(vitpose_cols) == len(VITPOSE_KEYPOINTS_NAMES)*3), f"Expected {len(VITPOSE_KEYPOINTS_NAMES)*3} ViTPose columns in the include columns, got {len(vitpose_cols)}"
        
        if hf_local_dir == "default":
            hf_local_dir = os.path.join(here, "..", "datasets", "hf_data")
        
        if not os.path.exists(hf_local_dir):
            prWarning(f"HF local directory {hf_local_dir} does not exist, creating it")
            os.makedirs(hf_local_dir)
        
        # Download the dataset
        prInfo(f"Downloading/loading dataset from {hf_dataset_revision} revision")
        
        if self.offline_mode:
            prInfo("Offline mode enabled - will not download from Hugging Face")
            prWarning("Offline mode used : dataset may not be up to date")
            assert(os.path.exists(hf_local_dir)), f"HF local directory {hf_local_dir} does not exist"
            
            # classic dataset
            if hf_dataset_revision == "main":
                # new dataset
                dataset_csvs = MAIN_DATASET_FILENAMES
            elif hf_dataset_revision == "3c8a342548534b6b92d32b0099e266962facdf45":
                # legacy dataset
                dataset_csvs = LEGACY_DATASET_FILENAMES
            else:
                raise ValueError(f"Invalid Hugging Face revision: {hf_dataset_revision}")
                
            dataset_csvs.sort()
            hf_data_dir = hf_local_dir
        
            for csv in dataset_csvs:
                assert os.path.exists(os.path.join(hf_local_dir, csv)), f"File {csv} does not exist in {hf_local_dir}"
            
            if self.do_recenter_interaction_zone:
                assert os.path.exists(os.path.join(hf_local_dir, "interaction_zone_center_positions.json")), f"File interaction_zone_center_positions.json does not exist in {hf_local_dir}"
                self.interaction_zone_center_positions = read_json_to_dic(os.path.join(hf_local_dir, "interaction_zone_center_positions.json"))

        else:
            repo_id = "rlorlou/HUI360"            
            # dry run to get the infos about the files (the actual current valid list of files in main)
            self.dry_run_infos = snapshot_download(repo_id=repo_id, 
                                            repo_type="dataset", 
                                            ignore_patterns="ignore.json",
                                            revision = hf_dataset_revision,
                                            local_dir = hf_local_dir,
                                            max_workers=self.num_workers if self.num_workers > 0 else 8,
                                            dry_run=True)
            all_csvs_size, cached_csvs_size, will_download_csvs_size = self.print_dry_run_infos()
            if will_download_csvs_size > 0 and not allow_download:
                answer = input(f"Will download {will_download_csvs_size/1e9:.2f} GB of csv files, continue ? (y/n)")
                if answer != "y":
                    prError("Exiting...")
                    exit()
            
            dataset_csvs = [file_info.filename for file_info in self.dry_run_infos if file_info.filename.endswith(".csv")]
            dataset_csvs.sort()
            
            if will_download_csvs_size > 0:
                # actually download / fetch in cache
                hf_data_dir = snapshot_download(repo_id=repo_id, 
                                                repo_type="dataset", 
                                                ignore_patterns="ignore.json",
                                                revision = hf_dataset_revision,
                                                local_dir = hf_local_dir,
                                                max_workers=self.num_workers if self.num_workers > 0 else 8,
                                                dry_run=False)
            else:
                hf_data_dir = hf_local_dir
            
            print(f"\n\nLocal path to csv files: {hf_data_dir}")
            
            if self.do_recenter_interaction_zone:
                # make sure to download the interaction zone center positions file if necessary, put it along the csv files
                hf_hub_download(repo_id="rlorlou/HUI360", 
                                filename = "interaction_zone_center_positions.json",
                                repo_type="dataset", 
                                revision = "main",
                                local_dir = hf_local_dir,
                                )
                
                prInfo(f"Downloaded interaction zone center positions file to {os.path.join(hf_local_dir, 'interaction_zone_center_positions.json')}")
                
                self.interaction_zone_center_positions = read_json_to_dic(os.path.join(hf_local_dir, "interaction_zone_center_positions.json"))
            
        # Load the dataset, recording by recording with some filtering using multiprocessing
        keep_recordings = []
        existing_recordings = []

        # Prepare arg list to avoid closure
        arg_list = []
        for csv in dataset_csvs:
            arg_list.append((csv, hf_data_dir, include_recordings, include_columns, return_masks, self.verbose))

        results = []
        mp_ctx = mp.get_context("fork" if hasattr(os, 'fork') else "spawn")
        workers = self.num_workers if self.num_workers > 0 else int(mp.cpu_count()/2)
        with mp_ctx.Pool(processes=min(workers, len(arg_list))) as pool:
            results = pool.map(process_csv, arg_list)

        for recording_dataset, csv_recording in results:
            existing_recordings.append(csv_recording)
            if recording_dataset is not None:
                keep_recordings.append(recording_dataset)
            # else is filtered out

            
        assert len(keep_recordings) > 0, f"No recordings found ! (Existing recordings : {existing_recordings}, requested {include_recordings})"
        
        self.dataset = pd.concat(keep_recordings)
        self.metadata_columns_in_dataset = self.get_metadata_columns_in_dataset()
        self.data_columns_in_dataset = self.get_data_columns_in_dataset()

        if include_recordings != "all":
            self.log(f"Filtered by recordings and got {len(keep_recordings)} recordings out of {len(existing_recordings)} existing recordings", "success")
            if len(keep_recordings) != len(include_recordings):
                self.log(f"Requested {len(include_recordings)} recordings but got {len(keep_recordings)} recording", "warning")
        
        # Filter tracks if needed
        if self.include_tracks != "all":
            unique_tracks_before = self.dataset["unique_track_identifier"].unique() 
            self.dataset = self.dataset[self.dataset["unique_track_identifier"].isin(self.include_tracks)]
            unique_tracks_after = self.dataset["unique_track_identifier"].unique()
            self.log(f"Filtered by tracks and kept {len(unique_tracks_after)} tracks out of {len(unique_tracks_before)} (requested {len(self.include_tracks)} tracks)", "success")
            if len(unique_tracks_after) != len(self.include_tracks):
                self.log(f"Requested {len(self.include_tracks)} tracks but got {len(unique_tracks_after)} tracks", "warning")
        else:
            self.log(f"No tracks filtering", "success")


        tic = time.time()

        # Filter track that will not be able to be used as input
        self.input_by_tracks = {} # {unique_track_identifier: [(possible_start_frame, number_of_frames), ...], ...} # such that given a track we can sample a random possible input, frames have to be indexed in track_data
        unique_tracks_identifiers = self.dataset["unique_track_identifier"].unique()
        
        # prepare input for multiprocessing
        self.log(f"Preparing inputs for multiprocessing of {len(unique_tracks_identifiers)} tracks (process input and find possible starting points)", "info")
        process_track_input_args = []
        self.datasets_by_unique_track_identifier = dict(tuple(self.dataset.groupby('unique_track_identifier'))) # {unique_track_identifier: track_data, ...}, track_data are already sorted by image_index
        for unique_track_identifier in unique_tracks_identifiers:
            process_track_input_args.append({"unique_track_identifier": unique_track_identifier, 
                                             "track_data": self.datasets_by_unique_track_identifier[unique_track_identifier],
                                             "input_length_in_frames": self.input_length_in_frames,
                                             "fixed_input_length": self.fixed_input_length,
                                             "min_length_in_frames": self.min_length_in_frames,
                                             "max_length_in_frames": self.max_length_in_frames,
                                             "interaction_cutoff": self.interaction_cutoff,
                                             "positive_cutoff": self.positive_cutoff,
                                             "force_positive_samples": self.force_positive_samples,
                                             "force_aligment_with_biggest_mask_size": self.force_aligment_with_biggest_mask_size,
                                             "additional_filtering_dict": self.additional_filtering_dict, 
                                             "min_keypoints_filter": self.min_keypoints_filter,
                                             "center_on_onset": self.center_on_onset,
                                             "cutoffs_filtering": self.cutoffs_filtering})
            
        self.unique_track_to_first_interaction_index = {}

        # Use torch multiprocessing for parallelization
        self.log(f"Running with torch multiprocessing for parallelization of {len(unique_tracks_identifiers)} tracks (process input and find possible starting points)", "info")
        workers = self.num_workers if self.num_workers > 0 else int(mp.cpu_count()/2)
        with mp_ctx.Pool(processes=min(workers, len(unique_tracks_identifiers))) as pool:
            track_results = pool.map(process_track_input, process_track_input_args)


        # in the case of a single input per track (i.e. inputs_per_track_stride == -1) and if using fix_index_per_track
        # define the index to choose for each track at each epoch when calling __getitem__
        # NB : self.index_choice_by_unique_track_identifier is unused when inputs_per_track_stride > 0
        self.index_choice_by_unique_track_identifier = {}
        for unique_track_iter, (unique_track, first_interaction_index, possible_indices) in enumerate(track_results):
            self.unique_track_to_first_interaction_index[unique_track] = first_interaction_index
            
            if len(possible_indices) > 0:
                self.input_by_tracks[unique_track] = possible_indices
                if self.fix_index_per_track:
                    if fix_index_per_track_list is not None:
                        # this fix_index_per_track_list can be given to provide reproducibility from a run to another
                        self.index_choice_by_unique_track_identifier[unique_track] = fix_index_per_track_list[unique_track_iter%len(fix_index_per_track_list)]%len(possible_indices)
                    else:
                        # or we choose one index per track at the dataset initialization time, and then this will be used for each epoch
                        self.index_choice_by_unique_track_identifier[unique_track] = torch.randint(0, len(possible_indices), (1,)).item()

                else:
                    # if not fixed, then we will choose a random index at each epoch when calling __getitem__
                    self.index_choice_by_unique_track_identifier[unique_track] = None
        
        
        segments_labels = [] # just for counting when using a stride
        if self.inputs_per_track_stride > 0:
            self.idx_to_unique_track_identifier = []
            self.idx_to_index_among_track = []
            self.idx_to_label = []
            for unique_track, inputs in self.input_by_tracks.items():
                for i in range(0, len(inputs), self.inputs_per_track_stride):
                    self.idx_to_unique_track_identifier.append(unique_track)
                    segments_labels.append(inputs[i][2])
                    self.idx_to_index_among_track.append(i)
        else:
            self.idx_to_label = []
            self.idx_to_unique_track_identifier = list(self.input_by_tracks.keys()) # to query a track when using __getitem__
            for unique_track, inputs in self.input_by_tracks.items():
                unique_track_possible_labels = [inputs[i][2] for i in range(0, len(inputs))]
                has_positive_label = any(label == 1 for label in unique_track_possible_labels)
                if has_positive_label:
                    self.idx_to_label.append(1)
                else:
                    self.idx_to_label.append(0)
            self.idx_to_index_among_track = None # if no stride index_choice_by_unique_track_identifier (or just random choice) will be used in __getitem__
            
        # Do some counting
        total_possible_segments = 0
        total_possible_segments_with_interaction = 0
        total_tracks = 0
        total_tracks_with_possible_interaction = 0
        tracks_with_possible_interaction = []
        tracks_with_valid_input = []
        for unique_track, possible_indices in self.input_by_tracks.items():
            total_possible_inputs_for_this_track = len(possible_indices)
            if len(possible_indices) == 0:
                continue

            tracks_with_valid_input.append(unique_track)
            total_possible_inputs_with_interaction_for_this_track = sum(1 for _, _, label in possible_indices if label == 1)
            total_tracks += 1
            
            if total_possible_inputs_with_interaction_for_this_track > 0 and unique_track not in tracks_with_possible_interaction:
                tracks_with_possible_interaction.append(unique_track)
                total_tracks_with_possible_interaction += 1
                
            total_possible_segments += total_possible_inputs_for_this_track
            total_possible_segments_with_interaction += total_possible_inputs_with_interaction_for_this_track

        self.total_inputs = total_possible_segments
        
        self.total_positives_tracks = total_tracks_with_possible_interaction
        self.total_negatives_tracks = total_tracks - total_tracks_with_possible_interaction
        self.total_possible_positives_segments = total_possible_segments_with_interaction # possible positives !
        self.total_possible_negatives_segments = total_possible_segments - total_possible_segments_with_interaction # possible negatives !
        
        tac = time.time()
        
        # prTimer(f"Total possible segments computation", tic, tac)
        self.log(f"Total possible segments: {total_possible_segments} among {total_tracks} tracks", "success")
        self.log(f"Total possible segments with interaction: {total_possible_segments_with_interaction} among {total_tracks_with_possible_interaction} tracks", "success")
        
        if self.inputs_per_track_stride != -1:
            self.total_used_positive_segments = sum(1 for label in segments_labels if label == 1)
            self.total_used_negative_segments = sum(1 for label in segments_labels if label == 0)
            self.log(f"Total used positive segments: {self.total_used_positive_segments} among {total_possible_segments_with_interaction} possible positive segments", "success")
            self.log(f"Total used negative segments: {self.total_used_negative_segments} among {total_possible_segments - total_possible_segments_with_interaction} possible negative segments", "success")
        else:
            self.total_used_positive_segments = None
            self.total_used_negative_segments = None
            
        #### Write to a file the list of tracks with valid input ####
        # timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        # save_tracks_path = os.path.join(here, "..", "utils", "tests", f"{timestamp}_tracks_with_valid_input_len_{self.input_length_in_frames}_int_{self.interaction_cutoff}_pos_{self.positive_cutoff}_force_pos_{self.force_positive_samples}_force_align_{self.force_aligment_with_biggest_mask_size}.txt")
        # with open(save_tracks_path, "w") as f:
        #     for track in tracks_with_valid_input:
        #         f.write(f"{track}\n")
        # prDebug(f"Saved tracks with valid input to file {save_tracks_path}")
        
        # save_track_with_possible_interaction_path = os.path.join(here, "..", "utils", "tests", f"{timestamp}_tracks_with_possible_interaction_len_{self.input_length_in_frames}_int_{self.interaction_cutoff}_pos_{self.positive_cutoff}_force_pos_{self.force_positive_samples}_force_align_{self.force_aligment_with_biggest_mask_size}.txt")
        # with open(save_track_with_possible_interaction_path, "w") as f:
        #     for track in tracks_with_possible_interaction:
        #         f.write(f"{track}\n")
        # prDebug(f"Saved tracks with possible interaction to file {save_track_with_possible_interaction_path}")
        
        #######################################################################


    def print_dry_run_infos(self):
        print(f"\n\nHF dry-run got infos about {len(self.dry_run_infos)} files")
        all_csvs = []
        all_csvs_size = 0
        cached_csvs = []
        cached_csvs_size = 0
        will_download_csvs = []
        will_download_csvs_size = 0
        for file_info in self.dry_run_infos:
            if file_info.filename.endswith(".csv"):
                all_csvs.append(file_info.filename)
                all_csvs_size += file_info.file_size
                if file_info.is_cached:
                    cached_csvs.append(file_info.filename)
                    cached_csvs_size += file_info.file_size
                elif file_info.will_download:
                    will_download_csvs.append(file_info.filename)
                    will_download_csvs_size += file_info.file_size
                
        prInfo(f"Total csv files found in HF dataset: {len(all_csvs)} ({all_csvs_size/1e9:.2f} GB)")
        prInfo(f"Total csv files already cached: {len(cached_csvs)} ({cached_csvs_size/1e9:.2f} GB)")
        prInfo(f"Total csv files will be downloaded: {len(will_download_csvs)} ({will_download_csvs_size/1e9:.2f} GB)")
        return all_csvs_size, cached_csvs_size, will_download_csvs_size
    
    def log(self, text: str, level: str = "info"):
        if self.verbose:
            if level == "info":
                prInfo(text)
            elif level == "warning":
                prWarning(text)
            elif level == "error":
                prError(text)
            elif level == "debug":
                prDebug(text)
            elif level == "success":
                prSuccess(text)
                
    def get_metadata_columns_in_dataset(self) -> list:
        """ Get the metadata columns in the dataset
        
        Returns:
            list: list of metadata columns
        """
        return [col for col in self.dataset.columns if col in self.metadata_columns]
    
    def get_data_columns_in_dataset(self) -> list:
        """ Get the data columns in the dataset
        
        Returns:
            list: list of data columns
        """
        data_columns = [col for col in self.dataset.columns if col not in self.metadata_columns]
        return data_columns
    

    def normalize_data(self, input_df: pd.DataFrame, normalize_bbox_df: pd.DataFrame, image_height: int, image_width: int) -> pd.DataFrame:
        """ Normalize the input data. Normalize box in image coordinates to [0, 1] and normalize keypoints either in image coordinates or in box coordinates to [0, 1]. Normalize mask size with max and min.
        
        Args:
            input_df: input data (len(input_df) = number of frames)
            normalize_bbox_df: dataframe with the bounding box (in image pixels) coordinates to normalize
            image_height: image height
            image_width: image width
        
        Returns:
            pd.DataFrame: normalized input data (len(input_df) = number of frames)
        """
        
        
        # if "xmin" in input_df.columns and "ymin" in input_df.columns and "xmax" in input_df.columns and "ymax" in input_df.columns:
        
        if self.normalize_keypoints_in_track == "norm_xy" or self.normalize_keypoints_in_track == "norm_x":
            # make a fix so that when we normalize the keypoints in the track (along x) we handle the 360 wrapping correctly
            # to do so we will express all x coordinates relative to the first element of the segment
            xcenters_values = normalize_bbox_df["xmin_meta"].values + (normalize_bbox_df["xmax_meta"].values - normalize_bbox_df["xmin_meta"].values) / 2
            first_x_center = xcenters_values[0]
            offsets_to_previous = []
            offsets_to_apply = [0] #int(-first_x_center)]
            for i in range(1,len(normalize_bbox_df)):
                diff_x = xcenters_values[i] - xcenters_values[i-1]
                if diff_x > image_width/2:
                    diff_x = diff_x - image_width
                elif diff_x < -image_width/2:
                    diff_x = diff_x + image_width
                offsets_to_previous.append(int(diff_x))
                cumulated_diff = sum(offsets_to_previous)
                offset_to_apply = -xcenters_values[i] + first_x_center + cumulated_diff
                offsets_to_apply.append(int(offset_to_apply))
            
            
            # print("================================================")
            # print([int(f) for f in xcenters_values])
            # print([int(f) for f in offsets_to_previous])
            # print([int(f) for f in offsets_to_apply])
            # print([int(f+v) for f, v in zip(xcenters_values, offsets_to_apply)])
            # print("================================================")
            
            # additional if we are completely wrapped around, lets shift everything by image_width to be in the normal range
            new_center_positions = [int(f+v) for f, v in zip(xcenters_values, offsets_to_apply)]
            if np.all(np.array(new_center_positions) > image_width):
                offsets_to_apply = [int(v-image_width) for v in offsets_to_apply]
            
            normalize_bbox_df.loc[:, "xmin_meta"] = normalize_bbox_df["xmin_meta"] + offsets_to_apply
            # print("new normalize xmin_meta: ", normalize_bbox_df["xmin_meta"].values)
            normalize_bbox_df.loc[:, "xmax_meta"] = normalize_bbox_df["xmax_meta"] + offsets_to_apply
            # print("new normalize xmax_meta: ", normalize_bbox_df["xmax_meta"].values)
            
            if "xmin" in input_df.columns:
                input_df.loc[:, "xmin"] = input_df["xmin"] + offsets_to_apply
            if "xmax" in input_df.columns:
                input_df.loc[:, "xmax"] = input_df["xmax"] + offsets_to_apply
            
        for col in input_df.columns:
            if col.startswith("vitpose_") or col.startswith("sapiens_"):
                if self.normalize_keypoints_in_box:
                    if col.endswith("_x"):
                        input_df[col] = (input_df[col] - normalize_bbox_df["xmin_meta"]) / (normalize_bbox_df["xmax_meta"] - normalize_bbox_df["xmin_meta"])
                    elif col.endswith("_y"):
                        input_df[col] = (input_df[col] - normalize_bbox_df["ymin_meta"]) / (normalize_bbox_df["ymax_meta"] - normalize_bbox_df["ymin_meta"])
                elif self.normalize_keypoints_in_track == "norm_xy":
                                                                                                
                    normalize_track_xmin = normalize_bbox_df["xmin_meta"].values.min() # overall min/max on the segment
                    normalize_track_xmax = normalize_bbox_df["xmax_meta"].values.max() # overall min/max on the segment
                    normalize_track_ymin = normalize_bbox_df["ymin_meta"].values.min() # overall min/max on the segment
                    normalize_track_ymax = normalize_bbox_df["ymax_meta"].values.max() # overall min/max on the segment
                    if col.endswith("_x"):
                        input_df.loc[:, col] = input_df[col] + offsets_to_apply
                        input_df[col] = (input_df[col] - normalize_track_xmin) / (normalize_track_xmax - normalize_track_xmin)
                    elif col.endswith("_y"):
                        input_df[col] = (input_df[col] - normalize_track_ymin) / (normalize_track_ymax - normalize_track_ymin)
                        
                elif self.normalize_keypoints_in_track == "norm_x":
                    normalize_track_xmin = normalize_bbox_df["xmin_meta"].values.min() # overall min/max on the segment
                    normalize_track_xmax = normalize_bbox_df["xmax_meta"].values.max() # overall min/max on the segment
                    if col.endswith("_x"):
                        input_df.loc[:, col] = input_df[col] + offsets_to_apply
                        input_df[col] = (input_df[col] - normalize_track_xmin) / (normalize_track_xmax - normalize_track_xmin)
                    elif col.endswith("_y"):
                        input_df[col] = input_df[col] / image_height
                elif self.normalize_keypoints_in_track == "norm_y":
                    normalize_track_ymin = normalize_bbox_df["ymin_meta"].values.min() # overall min/max on the segment
                    normalize_track_ymax = normalize_bbox_df["ymax_meta"].values.max() # overall min/max on the segment
                    if col.endswith("_x"):
                        input_df[col] = input_df[col] / image_width
                    elif col.endswith("_y"):
                        input_df[col] = (input_df[col] - normalize_track_ymin) / (normalize_track_ymax - normalize_track_ymin)
                else:
                    if col.endswith("_x"):
                        input_df[col] = input_df[col] / image_width
                    elif col.endswith("_y"):
                        input_df[col] = input_df[col] / image_height

            #input_df["xmin"] = input_df["xmin"] / image_width
            #input_df["ymin"] = input_df["ymin"] / image_height
            #input_df["xmax"] = input_df["xmax"] / image_width
            #input_df["ymax"] = input_df["ymax"] / image_height
            
        if "xmin" in input_df.columns:
            input_df["xmin"] = input_df["xmin"] / image_width
        if "xmax" in input_df.columns:
            input_df["xmax"] = input_df["xmax"] / image_width
        if "ymin" in input_df.columns:
            input_df["ymin"] = input_df["ymin"] / image_height
        if "ymax" in input_df.columns:
            input_df["ymax"] = input_df["ymax"] / image_height
        
        # if "mask_size" in input_df.columns and "mask_size" in self.standardize_data:
        #     input_df["mask_size"] = (input_df["mask_size"] - self.mean_mask_size) / self.std_mask_size
        
        if self.standardize_data == "all":
            for col in input_df.columns:
                if col == "xmin" or col == "xmax" or col == "ymin" or col == "ymax":
                    # simple case the stastics are already computed for xmin, xmax, ymin, ymax after divding by image width and height
                    input_df[col] = (input_df[col] - HUI_NORMALIZATION_VALUES[col]["mean"]) / HUI_NORMALIZATION_VALUES[col]["std"]
                elif col == "mask_size":
                    # simple case the stastics are already computed on raw number of pixels
                    input_df[col] = (input_df[col] - HUI_NORMALIZATION_VALUES[col]["mean"]) / HUI_NORMALIZATION_VALUES[col]["std"]
                    
                elif (col.startswith("vitpose_") or col.startswith("sapiens_")):
                    
                    if self.normalize_keypoints_in_box:
                        # we womm use _box_norm for keypoints in box coordinates
                        if col.endswith("_x"):
                            input_df[col] = (input_df[col] - HUI_NORMALIZATION_VALUES[col+"_box_norm"]["mean"]) / HUI_NORMALIZATION_VALUES[col+"_box_norm"]["std"]
                        elif col.endswith("_y"):
                            input_df[col] = (input_df[col] - HUI_NORMALIZATION_VALUES[col+"_box_norm"]["mean"]) / HUI_NORMALIZATION_VALUES[col+"_box_norm"]["std"]
                    
                    elif self.normalize_keypoints_in_track != "none":
                        # use image_norm values for now... TODO : change this !
                        if col.endswith("_x"):
                            input_df[col] = (input_df[col] - HUI_NORMALIZATION_VALUES[col+"_image_norm"]["mean"]) / HUI_NORMALIZATION_VALUES[col+"_image_norm"]["std"]
                        elif col.endswith("_y"):
                            input_df[col] = (input_df[col] - HUI_NORMALIZATION_VALUES[col+"_image_norm"]["mean"]) / HUI_NORMALIZATION_VALUES[col+"_image_norm"]["std"]
                                            
                    else: # not self.normalize_keypoints_in_box and not self.normalize_keypoints_in_track:
                        # we womm use _image_norm for keypoints in image coordinates
                        if col.endswith("_x"):
                            input_df[col] = (input_df[col] - HUI_NORMALIZATION_VALUES[col+"_image_norm"]["mean"]) / HUI_NORMALIZATION_VALUES[col+"_image_norm"]["std"]
                        elif col.endswith("_y"):
                            input_df[col] = (input_df[col] - HUI_NORMALIZATION_VALUES[col+"_image_norm"]["mean"]) / HUI_NORMALIZATION_VALUES[col+"_image_norm"]["std"]
                        
        elif self.standardize_data == "mask_only":
            if "mask_size" in input_df.columns:
                input_df["mask_size"] = (input_df["mask_size"] - HUI_NORMALIZATION_VALUES["mask_size"]["mean"]) / HUI_NORMALIZATION_VALUES["mask_size"]["std"]
                
        return input_df

    def flip_data(self, input_df: pd.DataFrame, normalize_bbox_df: pd.DataFrame, image_height: int, image_width: int) -> pd.DataFrame:
        """ Flip the data horizontally, also invert the left and right keypoints indexes
        
        To apply before normalizing the data, warning : this may result in negative coordinates for keypoints and boxes !
        
        Args:
            input_df: input data (len(input_df) = number of frames)
            normalize_bbox_df: dataframe with the bounding box (in image pixels) coordinates to normalize
            image_height: image height
            image_width: image width
        
        Returns:
            pd.DataFrame: flipped input data (len(input_df) = number of frames)
            pd.DataFrame: flipped normalize_bbox_df
        """
        
        new_normalize_bbox_df = normalize_bbox_df.copy()
        new_normalize_bbox_df.loc[:, "xmin_meta"] = image_width - normalize_bbox_df["xmax_meta"]
        new_normalize_bbox_df.loc[:, "xmax_meta"] = image_width - normalize_bbox_df["xmin_meta"]
        
        new_input_df = input_df.copy()
        if "xmin" in new_input_df.columns:
            new_input_df.loc[:, "xmin"] = image_width - input_df["xmax"]
        if "xmax" in new_input_df.columns:
            new_input_df.loc[:, "xmax"] = image_width - input_df["xmin"]
            
        for col in input_df.columns:
            if col.startswith("vitpose_") or col.startswith("sapiens_"):
                # First find the new column name to put the data in
                if "left" in col:
                    new_col_name = col.replace("left", "right")
                elif "right" in col:
                    new_col_name = col.replace("right", "left")
                else:
                    # keypoints that are centered (e.g. neck, nose etc.)
                    new_col_name = col
                
                # Add x data with flipping
                if col.endswith("_x"):
                    new_input_df.loc[:, new_col_name] = image_width - input_df[col]
                
                # Add y data as original
                elif col.endswith("_y"):
                    new_input_df.loc[:, new_col_name] = input_df[col]
        
        return new_input_df, new_normalize_bbox_df

    def jitter_data(self, input_df: pd.DataFrame, normalize_bbox_df: pd.DataFrame, image_height: int, image_width: int, jitter_x_pct: float = 0.0, jitter_y_pct: float = 0.0) -> pd.DataFrame:
        """ Jitter the data horizontally and vertically
        
        Args:
            input_df: input data (len(input_df) = number of frames)
            normalize_bbox_df: dataframe with the bounding box (in image pixels) coordinates to normalize
            image_height: image height
            image_width: image width
            jitter_x_pct: percentage of jitter in the x direction
            jitter_y_pct: percentage of jitter in the y direction
            
        Returns:
            pd.DataFrame: jittered input data (len(input_df) = number of frames)
            pd.DataFrame: jittered normalize_bbox_df
        """
        
        y_jitter = int(jitter_y_pct * image_height)
        x_jitter = int(jitter_x_pct * image_width)
        
        normalize_bbox_df.loc[:, "xmin_meta"] = normalize_bbox_df["xmin_meta"] + x_jitter
        normalize_bbox_df.loc[:, "xmax_meta"] = normalize_bbox_df["xmax_meta"] + x_jitter
        normalize_bbox_df.loc[:, "ymin_meta"] = normalize_bbox_df["ymin_meta"] + y_jitter
        normalize_bbox_df.loc[:, "ymax_meta"] = normalize_bbox_df["ymax_meta"] + y_jitter
        
        if "xmin" in input_df.columns:
            input_df.loc[:, "xmin"] = input_df["xmin"] + x_jitter
        if "xmax" in input_df.columns:
            input_df.loc[:, "xmax"] = input_df["xmax"] + x_jitter
        if "ymin" in input_df.columns:
            input_df.loc[:, "ymin"] = input_df["ymin"] + y_jitter
        if "ymax" in input_df.columns:
            input_df.loc[:, "ymax"] = input_df["ymax"] + y_jitter
        
        for col in input_df.columns:
            if col.startswith("vitpose_") or col.startswith("sapiens_"):
                if col.endswith("_x"):
                    input_df.loc[:, col] = input_df[col] + x_jitter
                elif col.endswith("_y"):
                    input_df.loc[:, col] = input_df[col] + y_jitter
        
        return input_df, normalize_bbox_df
    
    def fix_keypoints_outside_box(self, input_df: pd.DataFrame, normalize_bbox_df: pd.DataFrame, image_width: int) -> pd.DataFrame:
        """ Fix the keypoints that are outside the box, boxes sometimes are over image_width (with wrapping) but keypoints keep the exact pixel coordinates
        Put the keypoints in the box even if it means putting them with _x > image_width
        
        Args:
            input_df: input data (len(input_df) = number of frames)
            normalize_bbox_df: dataframe with the bounding box (in image pixels) coordinates to normalize
        
        Returns:
            pd.DataFrame: fixed input data (len(input_df) = number of frames)
        """
        for col in input_df.columns:
            if (col.startswith("vitpose_") or col.startswith("sapiens_")) and col.endswith("_x"):
                diff_to_xmin = np.abs(input_df[col] - normalize_bbox_df["xmin_meta"])
                # a keypoint could be a bit outside of the box but not so much
                diff_to_xmin_mask = diff_to_xmin > image_width/2
                if diff_to_xmin_mask.sum() > 0:
                    # prWarning(f"Found {diff_to_xmin_mask.sum()} rows where {col} is more than {image_width/2} pixels away from xmin_meta, applied shift to them")
                    input_df.loc[diff_to_xmin_mask, col] = input_df.loc[diff_to_xmin_mask, col] + image_width
        
        return input_df

    
    def recenter_interaction_zone(self, input_df: pd.DataFrame, 
                                  normalize_bbox_df: pd.DataFrame, 
                                  metadata: dict) -> (pd.DataFrame, pd.DataFrame):
        """ Recenter the interaction zone in the input data such that all coordinates are expressed relatively to an interaction zone 
        with horizontalcenter at image_width/2.
        
        Args:
            input_df: input data (len(input_df) = number of frames)
            normalize_bbox_df: dataframe with the bounding box (in image pixels) coordinates to normalize
            metadata: metadata dictionary
            
        Returns:
            pd.DataFrame: recentered input data (len(input_df) = number of frames)
            pd.DataFrame: recentered normalize_bbox_df
        """
        
        episode_name = f'{metadata["episode"]:04d}'
        recording_name = metadata["recording"]
        interaction_zone_center_positions = self.interaction_zone_center_positions[recording_name][episode_name]
        interaction_zone_center_position = np.array(interaction_zone_center_positions["interaction_zone_center_x"]).astype(np.int64)
        interaction_zone_center_position = interaction_zone_center_position[np.array(metadata["image_indexes"])]
        image_width = metadata["image_size"][0]
        shift_to_apply = (image_width/2 - interaction_zone_center_position).astype(np.int64)
        shift_applied = shift_to_apply.copy()
        
        normalize_bbox_df.loc[:, "xmin_meta"] = normalize_bbox_df["xmin_meta"] + shift_to_apply
        normalize_bbox_df.loc[:, "xmax_meta"] = normalize_bbox_df["xmax_meta"] + shift_to_apply

        # put back in the image if the shift made it go completely out of the image        
        xmin_over_mask = normalize_bbox_df["xmin_meta"] >= image_width # if the min is completely out of the image (over the right edge), then the max is also out of the image
        xmax_neg_mask = normalize_bbox_df["xmax_meta"] < 0 # if the max is completely out of the image (under the left edge), then the min is also out of the image
        
        normalize_bbox_df.loc[xmin_over_mask, "xmin_meta"] = normalize_bbox_df.loc[xmin_over_mask, "xmin_meta"] - image_width
        normalize_bbox_df.loc[xmin_over_mask, "xmax_meta"] = normalize_bbox_df.loc[xmin_over_mask, "xmax_meta"] - image_width
        
        normalize_bbox_df.loc[xmax_neg_mask, "xmin_meta"] = normalize_bbox_df.loc[xmax_neg_mask, "xmin_meta"] + image_width
        normalize_bbox_df.loc[xmax_neg_mask, "xmax_meta"] = normalize_bbox_df.loc[xmax_neg_mask, "xmax_meta"] + image_width

        shift_applied[xmin_over_mask] -= image_width
        shift_applied[xmax_neg_mask] += image_width
        
        if "xmin" in input_df.columns:
            assert("xmax" in input_df.columns), "xmax must be in input_df.columns if xmin is in input_df.columns"
            input_df.loc[:, "xmin"] = input_df["xmin"] + shift_to_apply
            input_df.loc[:, "xmax"] = input_df["xmax"] + shift_to_apply
            
            input_df.loc[xmin_over_mask, "xmin"] = input_df.loc[xmin_over_mask, "xmin"] - image_width
            input_df.loc[xmin_over_mask, "xmax"] = input_df.loc[xmin_over_mask, "xmax"] - image_width
            
            input_df.loc[xmax_neg_mask, "xmin"] = input_df.loc[xmax_neg_mask, "xmin"] + image_width
            input_df.loc[xmax_neg_mask, "xmax"] = input_df.loc[xmax_neg_mask, "xmax"] + image_width
        
        for col in input_df.columns:
            if (col.startswith("vitpose_") or col.startswith("sapiens_")) and col.endswith("_x"):
                input_df.loc[:, col] = input_df[col] + shift_to_apply
                input_df.loc[xmin_over_mask, col] = input_df.loc[xmin_over_mask, col] - image_width
                input_df.loc[xmax_neg_mask, col] = input_df.loc[xmax_neg_mask, col] + image_width
        
        
        # TODO
        
        return input_df, normalize_bbox_df, shift_applied, interaction_zone_center_position
        
        

    def recenter_boxes_keypoints_track(self, input_df: pd.DataFrame, normalize_bbox_df: pd.DataFrame, image_width: int) -> (pd.DataFrame, pd.DataFrame):
        """ Recenter the full track segment (preserving relative movement between frames in the segment)
        
        Args:
            input_df: input data (len(input_df) = number of frames)
            normalize_bbox_df: dataframe (length = number of frames) with the bounding box (in image pixels) coordinates to normalize
            image_width: image width
        Returns:
            pd.DataFrame: recentered input data (len(input_df) = number of frames)
            pd.DataFrame: recentered normalize_bbox_df
        """
        
        # TODO : take into account the wrapping of the equirectangular image, for now we assume the track is not wrapped around !
        
        track_xmin = normalize_bbox_df["xmin_meta"].values.min()
        track_xmax = normalize_bbox_df["xmax_meta"].values.max()
        track_width = track_xmax - track_xmin
        track_center = (track_xmin + track_xmax) / 2
        track_shift = (image_width/2 - track_center) # same value for every frame
        normalize_bbox_df.loc[:, "xmin_meta"] = (normalize_bbox_df["xmin_meta"].values + track_shift).astype(np.int64)
        normalize_bbox_df.loc[:, "xmax_meta"] = (normalize_bbox_df["xmax_meta"].values + track_shift).astype(np.int64)
        
        if "xmin" in input_df.columns:
            assert("xmax" in input_df.columns), "xmax must be in input_df.columns if xmin is in input_df.columns"
            input_df.loc[:, "xmin"] = input_df["xmin"] + track_shift
            input_df.loc[:, "xmax"] = input_df["xmax"] + track_shift
            
        for col in input_df.columns:
            if col.endswith("_x"):
                input_df.loc[:, col] = (input_df.loc[:, col] + track_shift).astype(np.int64)
                under_0_mask = input_df[col] < 0
                
                # should be corrected by fix_keypoints_outside_box
                if under_0_mask.sum() > 0:
                    prError(f"Found {under_0_mask.sum()} rows where {col} is less than 0, applied shift to them, it is not supposed to happen !")
                overbound_mask = input_df[col] > image_width
                if overbound_mask.sum() > 0:
                    prError(f"Found {overbound_mask.sum()} rows where {col} is greater than image_width, applied shift to them, it is not supposed to happen !")
                    
        # just a sanity check
        if "xmin" in input_df.columns:
            under_0_mask = input_df["xmin"] < 0
            if under_0_mask.sum() > 0:
                prError(f"Found {under_0_mask.sum()} rows where xmin is less than 0, applied shift to them, it is not supposed to happen !")
        if "xmax" in input_df.columns:
            overbound_mask = input_df["xmax"] > image_width
            if overbound_mask.sum() > 0:
                prError(f"Found {overbound_mask.sum()} rows where xmax is greater than image_width, applied shift to them, it is not supposed to happen !")
            
        return input_df, normalize_bbox_df
                
    
    def recenter_boxes_keypoints(self, input_df: pd.DataFrame, normalize_bbox_df: pd.DataFrame, image_width: int) -> (pd.DataFrame, pd.DataFrame):
        """ Recenter the boxes and keypoints in the input data.
        
        Args:
            input_df: input data (len(input_df) = number of frames)
            normalize_bbox_df: dataframe with the bounding box (in image pixels) coordinates to normalize
            image_width: image width
            by_track: if True, recenter the boxes and keypoints by track, if False, recenter the boxes and keypoints one by one
            
        Returns:
            pd.DataFrame: recentered input data (len(input_df) = number of frames)
            pd.DataFrame: recentered normalize_bbox_df
        """
    
        bbox_centers = (normalize_bbox_df["xmin_meta"] + normalize_bbox_df["xmax_meta"]) / 2        
        shift_to_apply = (image_width/2 - bbox_centers)

        normalize_bbox_df["xmin_meta"] = normalize_bbox_df["xmin_meta"] + shift_to_apply
        normalize_bbox_df["xmax_meta"] = normalize_bbox_df["xmax_meta"] + shift_to_apply
                    
        if "xmin" in input_df.columns:
            assert("xmax" in input_df.columns), "xmax must be in input_df.columns if xmin is in input_df.columns"
            input_df["xmin"] = input_df["xmin"] + shift_to_apply
            input_df["xmax"] = input_df["xmax"] + shift_to_apply
            # add image_width to those that fell under 0 -> should not be the case, all box coords are positives then the shift should not move them out of the frame
            under_0_mask = input_df["xmin"] < 0
            if under_0_mask.sum() > 0:
                prError(f"Found {under_0_mask.sum()} rows where xmin is less than 0, applied shift to them, it is not supposed to happen !")

        for col in input_df.columns:
            if col.endswith("_x"):
                input_df[col] = input_df[col] + shift_to_apply
                under_0_mask = input_df[col] < 0
                # should be corrected by fix_keypoints_outside_box
                if under_0_mask.sum() > 0:
                    prError(f"Found {under_0_mask.sum()} rows where {col} is less than 0, applied shift to them, it is not supposed to happen !")
                
        return input_df, normalize_bbox_df
    
    def reproject_to_perspective(self, input_df: pd.DataFrame, normalize_bbox_df: pd.DataFrame,
                                eq_height: int, eq_width: int) -> (pd.DataFrame, pd.DataFrame):
        """Reproject keypoints and bounding boxes from equirectangular to perspective coordinates.
        
        Converts pixel positions from an equirectangular image to a virtual perspective camera
        centered at (hcenter, vcenter) with given FOV. Points falling outside the perspective
        frame get coordinates (-1, -1) and confidence 0.0. The bounding box is recomputed from
        valid reprojected keypoints with a 10% margin.
        
        Args:
            input_df: input data with keypoint and optional bbox columns
            normalize_bbox_df: dataframe with bounding box columns for normalization
            eq_height: equirectangular image height
            eq_width: equirectangular image width
            
        Returns:
            (pd.DataFrame, pd.DataFrame): reprojected input_df and normalize_bbox_df
        """
        hfov_rad = np.radians(self.perspective_hfov)
        vfov_rad = np.radians(self.perspective_vfov)
        lon_c = np.radians(self.perspective_hcenter)
        lat_c = np.radians(self.perspective_vcenter)

        target_w = self.perspective_target_width
        target_h = self.perspective_target_height

        fx = (target_w / 2.0) / np.tan(hfov_rad / 2.0)
        fy = (target_h / 2.0) / np.tan(vfov_rad / 2.0)
        cx = target_w / 2.0
        cy = target_h / 2.0

        # Camera basis vectors in world coordinates (right-handed: X=right, Y=up, Z=forward)
        d = np.array([np.cos(lat_c) * np.sin(lon_c),
                       np.sin(lat_c),
                       np.cos(lat_c) * np.cos(lon_c)])
        right = np.array([np.cos(lon_c), 0.0, -np.sin(lon_c)])
        up = np.array([-np.sin(lon_c) * np.sin(lat_c),
                        np.cos(lat_c),
                        -np.cos(lon_c) * np.sin(lat_c)])

        new_input_df = input_df.copy()

        x_cols = [col for col in input_df.columns
                  if (col.startswith("vitpose_") or col.startswith("sapiens_")) and col.endswith("_x")]

        for x_col in x_cols:
            base_name = x_col[:-2]
            y_col = base_name + "_y"
            score_col = base_name + "_score"

            x_eq = input_df[x_col].values.astype(np.float64)
            y_eq = input_df[y_col].values.astype(np.float64)

            lon = (x_eq / eq_width) * 2.0 * np.pi - np.pi
            lat = np.pi / 2.0 - (y_eq / eq_height) * np.pi

            Xs = np.cos(lat) * np.sin(lon)
            Ys = np.sin(lat)
            Zs = np.cos(lat) * np.cos(lon)

            X_cam = right[0] * Xs + right[1] * Ys + right[2] * Zs
            Y_cam = up[0] * Xs + up[1] * Ys + up[2] * Zs
            Z_cam = d[0] * Xs + d[1] * Ys + d[2] * Zs

            in_front = Z_cam > 1e-6
            u = np.where(in_front, fx * X_cam / Z_cam + cx, -1.0)
            v = np.where(in_front, cy - fy * Y_cam / Z_cam, -1.0)

            in_bounds = in_front & (u >= 0) & (u < target_w) & (v >= 0) & (v < target_h)

            new_input_df[x_col] = np.where(in_bounds, u, -1.0)
            new_input_df[y_col] = np.where(in_bounds, v, -1.0)
            if score_col in input_df.columns:
                new_input_df[score_col] = np.where(in_bounds, input_df[score_col].values, 0.0)

        # Recompute bounding boxes from valid reprojected keypoints with 10% margin
        y_cols = [col[:-2] + "_y" for col in x_cols]
        if len(x_cols) > 0:
            x_vals = new_input_df[x_cols].values
            y_vals = new_input_df[y_cols].values

            valid_mask = (x_vals >= 0) & (y_vals >= 0)

            x_for_min = np.where(valid_mask, x_vals, np.inf)
            x_for_max = np.where(valid_mask, x_vals, -np.inf)
            y_for_min = np.where(valid_mask, y_vals, np.inf)
            y_for_max = np.where(valid_mask, y_vals, -np.inf)

            min_x = np.min(x_for_min, axis=1)
            max_x = np.max(x_for_max, axis=1)
            min_y = np.min(y_for_min, axis=1)
            max_y = np.max(y_for_max, axis=1)

            any_valid = valid_mask.any(axis=1)

            margin_x = np.where(any_valid, (max_x - min_x) * 0.1, 0.0)
            margin_y = np.where(any_valid, (max_y - min_y) * 0.1, 0.0)

            new_xmin = np.where(any_valid, np.maximum(0, min_x - margin_x), 0.0)
            new_xmax = np.where(any_valid, np.minimum(target_w, max_x + margin_x), float(target_w))
            new_ymin = np.where(any_valid, np.maximum(0, min_y - margin_y), 0.0)
            new_ymax = np.where(any_valid, np.minimum(target_h, max_y + margin_y), float(target_h))
        else:
            n = len(new_input_df)
            new_xmin = np.zeros(n)
            new_xmax = np.full(n, float(target_w))
            new_ymin = np.zeros(n)
            new_ymax = np.full(n, float(target_h))

        if "xmin" in new_input_df.columns:
            new_input_df["xmin"] = new_xmin
        if "xmax" in new_input_df.columns:
            new_input_df["xmax"] = new_xmax
        if "ymin" in new_input_df.columns:
            new_input_df["ymin"] = new_ymin
        if "ymax" in new_input_df.columns:
            new_input_df["ymax"] = new_ymax

        new_normalize_bbox_df = normalize_bbox_df.copy()
        new_normalize_bbox_df["xmin_meta"] = new_xmin
        new_normalize_bbox_df["xmax_meta"] = new_xmax
        new_normalize_bbox_df["ymin_meta"] = new_ymin
        new_normalize_bbox_df["ymax_meta"] = new_ymax

        return new_input_df, new_normalize_bbox_df

    def __len__(self):
        return len(self.idx_to_unique_track_identifier)

    def __getitem__(self, idx):
        
        tic = time.time()
        unique_track_identifier = self.idx_to_unique_track_identifier[idx]
        possible_inputs = self.input_by_tracks[unique_track_identifier]

        track_data = self.datasets_by_unique_track_identifier[unique_track_identifier]
        first_interaction_index = self.unique_track_to_first_interaction_index[unique_track_identifier]
        # track_data = track_data.sort_values(by="image_index") # already sorted by image_index
        if self.inputs_per_track_stride == -1:
            # only one input per track and per epoch
            if self.fix_index_per_track:
                # either fixed, i.e. always the same at each epoch...
                random_index = self.index_choice_by_unique_track_identifier[unique_track_identifier]
            else:
                # ... or random, will change at each epoch if the seed is not fixed
                random_index = torch.randint(0, len(possible_inputs), (1,)).item()
        else:
            # several inputs per track and per epoch (previously defined with the stride)
            random_index = self.idx_to_index_among_track[idx]
            
        input_index, number_of_frames, label = possible_inputs[random_index]

        image_indexes_track = range(input_index, input_index + number_of_frames, self.subsample_frames)
        # input_df = track_data.iloc[input_index:input_index + number_of_frames]
        # input_df = input_df.iloc[::self.subsample_frames] # apply stride to the input dataframe
        input_df = track_data.iloc[image_indexes_track]

        metadata_df = input_df[self.metadata_columns_in_dataset]

        metadata_dict = {}
        metadata_dict["unique_track_identifier"] = metadata_df["unique_track_identifier"].iloc[0]
        metadata_dict["recording"] = metadata_df["recording"].iloc[0]
        metadata_dict["episode"] = metadata_df["episode"].iloc[0]
        metadata_dict["image_size"] = (int(metadata_df["image_width"].iloc[0]), int(metadata_df["image_height"].iloc[0]))
        metadata_dict["image_indexes"] = [int(metadata_df["image_index"].iloc[i]) for i in range(len(metadata_df))]
        metadata_dict["image_indexes_track"] = [i for i in image_indexes_track]
        metadata_dict["engagements"] = [int(metadata_df["engagement"].iloc[i]) for i in range(len(metadata_df))]
        metadata_dict["time_to_first_interaction"] = metadata_df["time_to_first_interaction"].iloc[-1]
        metadata_dict["image_files"] = metadata_df["image_file"].tolist()
        metadata_dict["first_interaction_index"] = first_interaction_index
        metadata_dict["index_choice"] = random_index
        metadata_dict["input_index"] = input_index
        metadata_dict["number_of_frames"] = number_of_frames
        metadata_dict["xmin_meta"] = metadata_df["xmin_meta"].values
        metadata_dict["xmax_meta"] = metadata_df["xmax_meta"].values
        metadata_dict["ymin_meta"] = metadata_df["ymin_meta"].values
        metadata_dict["ymax_meta"] = metadata_df["ymax_meta"].values
        metadata_dict["time_to_interaction_by_frame"] = metadata_df["time_to_first_interaction"].values
        metadata_dict["orig_dataset"] = "HUI360"
        metadata_dict["file_path"] = "UnknownFilePath"

        image_height = metadata_df["image_height"].iloc[0]
        image_width = metadata_df["image_width"].iloc[0]
        
        input_df = input_df[self.data_columns_in_dataset]
        normalize_bbox_df = metadata_df[["xmin_meta", "xmax_meta", "ymin_meta", "ymax_meta"]]

        if self.do_fix_keypoints_outside_box:
            input_df = self.fix_keypoints_outside_box(input_df, normalize_bbox_df, image_width)
            
        if self.do_recenter_interaction_zone:
            input_df, normalize_bbox_df, shift_applied, interaction_zone_center_position = self.recenter_interaction_zone(input_df, normalize_bbox_df, metadata_dict)
            metadata_dict["shift_applied"] = [int(x) for x in shift_applied]
            metadata_dict["interaction_zone_center_position"] = [int(x) for x in interaction_zone_center_position]
            
        if self.random_jitter_position[0] > 0.0 or self.random_jitter_position[1] > 0.0:
            jitter_x_pct = (torch.rand(1).item() - 0.5) * 2 * self.random_jitter_position[0]
            jitter_y_pct = (torch.rand(1).item() - 0.5) * 2 * self.random_jitter_position[1]
            input_df, normalize_bbox_df = self.jitter_data(input_df, normalize_bbox_df, image_height, image_width, jitter_x_pct, jitter_y_pct)
        else:
            jitter_x_pct = 0.0
            jitter_y_pct = 0.0
            
        if self.random_flip_horizontal:
            flip_rand = torch.rand(1).item()
            if flip_rand < 0.5:
                flipped = True
                input_df, normalize_bbox_df = self.flip_data(input_df, normalize_bbox_df, image_height, image_width)
            else:
                flipped = False
        else:
            flipped = False
            
        metadata_dict["jitter_x_pct"] = jitter_x_pct
        metadata_dict["jitter_y_pct"] = jitter_y_pct
        metadata_dict["flipped"] = flipped

            
        # if self.normalize_in_image: # and "xmin" in self.data_columns_in_dataset and "ymin" in self.data_columns_in_dataset and "xmax" in self.data_columns_in_dataset and "ymax" in self.data_columns_in_dataset:
        if self.do_recentering:
            # input_df, normalize_bbox_df = self.recenter_boxes_keypoints(input_df, normalize_bbox_df, image_width)
            input_df, normalize_bbox_df = self.recenter_boxes_keypoints_track(input_df, normalize_bbox_df, image_width)

        DEBUG_PLOT = False
        if self.do_perspective_reprojection:
            if DEBUG_PLOT:
                # ---- DEBUG: snapshot before reprojection ----
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                from utils.data_utils import VITPOSE_KEYPOINTS_NAMES

                _COCO_LINKS = [
                    (0,1),(0,2),(1,3),(2,4),
                    (5,6),(5,7),(7,9),(6,8),(8,10),
                    (5,11),(6,12),(11,12),
                    (11,13),(13,15),(12,14),(14,16),
                ]

                def _extract_kps(df, frame_idx):
                    xs, ys, scores = [], [], []
                    for kp in VITPOSE_KEYPOINTS_NAMES:
                        xc = f"vitpose_{kp}_x"
                        yc = f"vitpose_{kp}_y"
                        sc = f"vitpose_{kp}_score"
                        xs.append(df[xc].iloc[frame_idx] if xc in df.columns else 0)
                        ys.append(df[yc].iloc[frame_idx] if yc in df.columns else 0)
                        scores.append(df[sc].iloc[frame_idx] if sc in df.columns else 0)
                    return np.array(xs), np.array(ys), np.array(scores)

                def _plot_skeleton(ax, xs, ys, scores, bbox, title, img_w, img_h):
                    valid = (xs >= 0) & (ys >= 0) & (scores > 0)
                    ax.scatter(xs[valid], ys[valid], c="red", s=30, zorder=5)
                    ax.scatter(xs[~valid], ys[~valid], c="gray", s=15, zorder=4, alpha=0.3)
                    for j1, j2 in _COCO_LINKS:
                        if valid[j1] and valid[j2]:
                            ax.plot([xs[j1], xs[j2]], [ys[j1], ys[j2]], "b-", lw=1.5, alpha=0.7)
                    for i, kp in enumerate(VITPOSE_KEYPOINTS_NAMES):
                        ax.text(xs[i], ys[i], kp, fontsize=5, alpha=0.7)
                    xmin, xmax, ymin, ymax = bbox
                    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                            linewidth=1.5, edgecolor="green", facecolor="none")
                    ax.add_patch(rect)
                    ax.set_xlim(-10, img_w + 10)
                    ax.set_ylim(img_h + 10, -10)
                    ax.set_aspect("equal")
                    ax.set_title(title, fontsize=8)

                n_frames = len(input_df)
                frames_to_show = [0] if n_frames == 1 else [0, n_frames // 2, n_frames - 1]
                frames_to_show = sorted(set(frames_to_show))

                pre_xs_list, pre_ys_list, pre_sc_list, pre_bbox_list = [], [], [], []
                for fi in frames_to_show:
                    xs, ys, sc = _extract_kps(input_df, fi)
                    pre_xs_list.append(xs); pre_ys_list.append(ys); pre_sc_list.append(sc)
                    pre_bbox_list.append((
                        normalize_bbox_df["xmin_meta"].iloc[fi],
                        normalize_bbox_df["xmax_meta"].iloc[fi],
                        normalize_bbox_df["ymin_meta"].iloc[fi],
                        normalize_bbox_df["ymax_meta"].iloc[fi],
                    ))
                eq_w, eq_h = image_width, image_height
                # ---- END snapshot ----

            input_df, normalize_bbox_df = self.reproject_to_perspective(input_df, normalize_bbox_df, image_height, image_width)
            image_width = self.perspective_target_width
            image_height = self.perspective_target_height
            metadata_dict["image_size"] = (image_width, image_height)
            metadata_dict["xmin_meta"] = normalize_bbox_df["xmin_meta"].values
            metadata_dict["xmax_meta"] = normalize_bbox_df["xmax_meta"].values
            metadata_dict["ymin_meta"] = normalize_bbox_df["ymin_meta"].values
            metadata_dict["ymax_meta"] = normalize_bbox_df["ymax_meta"].values
            metadata_dict["perspective_reprojected"] = True

            if DEBUG_PLOT:
                # ---- DEBUG: plot before & after ----
                n_cols = len(frames_to_show)
                fig, axs = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
                if n_cols == 1:
                    axs = axs.reshape(2, 1)

                for ci, fi in enumerate(frames_to_show):
                    _plot_skeleton(axs[0, ci], pre_xs_list[ci], pre_ys_list[ci], pre_sc_list[ci],
                                pre_bbox_list[ci],
                                f"BEFORE (equirect {eq_w}x{eq_h}) t={fi}",
                                eq_w, eq_h)

                    post_xs, post_ys, post_sc = _extract_kps(input_df, fi)
                    post_bbox = (
                        normalize_bbox_df["xmin_meta"].iloc[fi],
                        normalize_bbox_df["xmax_meta"].iloc[fi],
                        normalize_bbox_df["ymin_meta"].iloc[fi],
                        normalize_bbox_df["ymax_meta"].iloc[fi],
                    )
                    _plot_skeleton(axs[1, ci], post_xs, post_ys, post_sc,
                                post_bbox,
                                f"AFTER (persp {image_width}x{image_height}) t={fi}",
                                image_width, image_height)

                fig.suptitle(f"Perspective reprojection debug – {unique_track_identifier}", fontsize=10)
                plt.tight_layout()
                os.makedirs(os.path.join(here, "..", "debug_reprojection"), exist_ok=True)
                debug_path = os.path.join(here, "..", "debug_reprojection", f"{unique_track_identifier}.jpg")
                fig.savefig(debug_path, dpi=150)
                plt.close(fig)
                print(f"[DEBUG] Saved reprojection debug plot to {debug_path}")
                # ---- END DEBUG ----

        if self.normalize_in_image:
            input_df = self.normalize_data(input_df, normalize_bbox_df, image_height, image_width)
        
        # convert all columns to float32
        input_df = input_df.astype(np.float32)

        input_tensor = torch.tensor(input_df.values)
        label = torch.tensor(label)
        # prDebug(f"Input data: {input_tensor.shape}, label: {label}, metadata: {metadata_dict}")
        
        if self.format_by_channel:
            input_tensor = input_tensor_to_format_by_channel(input_tensor.unsqueeze(0), metadata_dict, self.data_columns_in_dataset, self.remove_joints)[0,...]
            
        tac = time.time()
        if self.verbose:
            prTimer(f"__getitem__ for {unique_track_identifier}", tic, tac)

        if self.return_images:
            episode = f"{metadata_df['episode'].iloc[0]:04d}"
            recording = metadata_df["recording"].iloc[0]
            image_files = metadata_df["image_file"].tolist()
            # print(f"episode: {episode}, recording: {recording}, image_files: {image_files}")
            images = []
            for image_file, xmin, xmax, ymin, ymax in zip(image_files, metadata_df["xmin_meta"], metadata_df["xmax_meta"], metadata_df["ymin_meta"], metadata_df["ymax_meta"]):
                image_path = get_image_path(self.raw_dataset_path, recording, episode, image_file)
                image = Image.open(image_path)
                image = F.pil_to_tensor(image) # type : torch.uint8, min : 0, max : 255, sum : 68950384060, device : cpu
                image = F.crop(image, int(ymin), int(xmin), int(ymax - ymin), int(xmax - xmin))
                image = self.image_transforms(image.unsqueeze(0)).squeeze(0)
                images.append(image)
                
            images_tensor = torch.stack(images) # shape (T, C, H, W)
        else:
            images_tensor = []
            
        if self.return_masks:
            masks_rle = metadata_df["mask_rle"]
            masks_rle = [[int(r) for r in rle_str.split(",")] for rle_str in masks_rle.values]
            masks_rle = [torch.tensor(x).to(device) for x in masks_rle]
            masks = [decode_RLE(mask_rle, (image_height, image_width)) for mask_rle in masks_rle] # list of masks tensor on device (1920, 3840), type bool
            transformed_masks = []
            for mask, xmin, xmax, ymin, ymax in zip(masks, metadata_df["xmin_meta"], metadata_df["xmax_meta"], metadata_df["ymin_meta"], metadata_df["ymax_meta"]):
                mask = F.crop(mask, int(ymin), int(xmin), int(ymax - ymin), int(xmax - xmin))
                mask = self.mask_transforms(mask.unsqueeze(0)).squeeze(0)
                transformed_masks.append(mask)
            masks_tensor = torch.stack(transformed_masks) # shape (T, H, W)
        else:
            masks_tensor = []
            
        return input_tensor, label, metadata_dict, images_tensor, masks_tensor