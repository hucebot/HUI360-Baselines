from math import e
from torch.utils.data import Dataset
import numpy as np
import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
from typing import Optional, Union
from huggingface_hub import snapshot_download

from utils.print_utils import *
from utils.rle_tools import *
from utils.data_utils import *

from datasets.hui_norm_values import HUI_NORMALIZATION_VALUES
import torch.multiprocessing as mp
import copy

def get_contiguous_true_segments(array: np.array) -> list:
    """ Get the contiguous true segments of a boolean array
    
    Example : [False,False,True,True,False,True,True,True] returns [(2, 2), (5, 3)]
    
    Args:
        array (np.array): boolean array
        
    Returns:
        list: list of tuples (start, length) with the positions of the true segments and their lengths. Inclusive of the first element.
    """
    
    assert(array.dtype == bool), "array must be a boolean array"
    assert(array.ndim == 1), "array must be a 1D array"
    assert(np.sum(array) > 0), "array must have at least one True"
    #Indices are found here
    segment_begin_positions = np.argwhere(np.diff(array,prepend=False,append=False)) # whenever it shifts, we add Falses, at both ends, since we are searching for Trues
    assert(len(segment_begin_positions) % 2 == 0), "array should always yield an even number of segment begin positions"
    #Conversion into list of 2-tuples
    segments_from_to = segment_begin_positions.reshape(len(segment_begin_positions)//2,2)
    segments_from_to = [tuple(r) for r in segments_from_to] # start position to start of the next segment (ie a segment of Falses)
    segments_from_length = [(start, to - start) for (start, to) in segments_from_to]
    return segments_from_length


def ssupaug_dataset_handling(recording_dataset: pd.DataFrame, 
                             recording_name: str, 
                             verbose: bool = True, 
                             add_sapiens_columns: bool = True,
                             filter_boxes: bool = True,
                             do_recentering: bool = False,
                             track_selection: list = []) -> pd.DataFrame:
    
    """ Special function to handle the ssupaug dataset for two possible issues : missing sapiens columns (not run yet), and recentering of boxes (some were outside of the frame)
    Also allow to perform track selection in advance to save memory
    """
    columns_in_recording_dataset = recording_dataset.columns

    if filter_boxes:
        # first drop rows if xmin is bigger than xmax (it should not happen but just in case)
        xmin_bigger_than_xmax_mask = recording_dataset["xmin"] > recording_dataset["xmax"]
        recording_dataset = recording_dataset[~xmin_bigger_than_xmax_mask]
        # prDebug(f"[{recording_name}] Dropping {xmin_bigger_than_xmax_mask.sum()/len(recording_dataset)*100:.2f}% rows where xmin is bigger than xmax")
        
        # then drop rows if width is more than half of the image width
        bbox_width_mask = recording_dataset["xmax"] - recording_dataset["xmin"] > (recording_dataset["image_width"]/2)
        recording_dataset = recording_dataset[~bbox_width_mask]
        # prDebug(f"[{recording_name}] Dropping {bbox_width_mask.sum()/len(recording_dataset)*100:.2f}% rows where width is more than half of the image width")


    if do_recentering:
        # Make that the bboxes are centered on the image width, this is better for distribution gap, but it removes all horizontal movement information
        
        bbox_centers = (recording_dataset["xmin"] + recording_dataset["xmax"]) / 2        
        shift_to_apply = (recording_dataset["image_width"]/2 - bbox_centers)
        
        # print("shift_to_apply : ", shift_to_apply.min(), shift_to_apply.max())
        # print("1. xmin : ", recording_dataset["xmin"].min(), recording_dataset["xmin"].max())
        # print("1. xmax : ", recording_dataset["xmax"].min(), recording_dataset["xmax"].max())
        
        recording_dataset["xmin"] = recording_dataset["xmin"] + shift_to_apply
        recording_dataset["xmax"] = recording_dataset["xmax"] + shift_to_apply
        
        # print("2. xmin : ", recording_dataset["xmin"].min(), recording_dataset["xmin"].max())
        # print("2. xmax : ", recording_dataset["xmax"].min(), recording_dataset["xmax"].max())
        
        # add image_width to thos that fell under 0
        under_0_mask = recording_dataset["xmin"] < 0
        recording_dataset.loc[under_0_mask, "xmin"] = recording_dataset.loc[under_0_mask, "xmin"] + recording_dataset.loc[under_0_mask, "image_width"]
        under_0_mask = recording_dataset["xmax"] < 0
        recording_dataset.loc[under_0_mask, "xmax"] = recording_dataset.loc[under_0_mask, "xmax"] + recording_dataset.loc[under_0_mask, "image_width"]
        
        # print("3. xmin : ", recording_dataset["xmin"].min(), recording_dataset["xmin"].max())
        # print("3. xmax : ", recording_dataset["xmax"].min(), recording_dataset["xmax"].max())
        
        for col in recording_dataset.columns:
            if col.endswith("_x"):
                recording_dataset[col] = recording_dataset[col] + shift_to_apply
                under_0_mask = recording_dataset[col] < 0
                recording_dataset.loc[under_0_mask, col] = recording_dataset.loc[under_0_mask, col] + recording_dataset.loc[under_0_mask, "image_width"]
    
    else:
        # make sure that the keypoints are well in the box
        
        image_width = recording_dataset["image_width"].iloc[0]
        xmax_bigger_than_image_width_mask = recording_dataset["xmax"] > image_width
        for col in recording_dataset.columns:
            if col.endswith("_x"):
                xkpt_smaller_than_half_image_width_mask = recording_dataset[col] < (image_width/2)
                # add the full image width if the two masks are True
                recording_dataset.loc[xkpt_smaller_than_half_image_width_mask & xmax_bigger_than_image_width_mask, col] += image_width
                
        
    if add_sapiens_columns:
        # check that there is 0 column starting with "sapiens"
        ncolumns_sapiens = len([col for col in columns_in_recording_dataset if col.startswith("sapiens")])
        # prDebug(f"[{recording_name}] Number of sapiens columns: {ncolumns_sapiens}")
        if ncolumns_sapiens == 308*3:
            # this is expected
            pass
        elif ncolumns_sapiens == 0:
            # prWarning(f"[{recording_name}] There is no sapiens columns, padding with 0s")
            #there is no sapiens columns, add fake ones
            sapiens_columns = [col for col in FULL_DATA_COLUMNS if col.startswith("sapiens")]
            fake_sapiens_cols = pd.DataFrame(np.zeros((len(recording_dataset), len(sapiens_columns))), columns=sapiens_columns, index=recording_dataset.index)
            recording_dataset = pd.concat([recording_dataset, fake_sapiens_cols], axis=1)
        else:
            raise ValueError(f"Unexpected number of sapiens columns: {len(ncolumns_sapiens)} (expected 308*3 or 0)")
    
    return recording_dataset

def process_csv(args):
    csv, hf_data_dir, include_recordings, include_columns, return_masks, verbose = args
    include_columns = copy.deepcopy(include_columns) # copy to avoid modifying the original list
    include_recordings = copy.deepcopy(include_recordings) # copy to avoid modifying the original list
    tic = time.time()
    csv_dataset = pd.read_csv(os.path.join(hf_data_dir, csv), nrows=1)
    csv_recording = csv_dataset["recording"].unique()[0]
    
    if verbose:
        prInfo(f"csv dataset {csv} - {csv_recording}")
    
    if include_recordings != "all":
        if csv_recording not in include_recordings:
            # prWarning(f"Skipping csv {csv} - {csv_recording}")
            return None, csv_recording

    recording_dataset = pd.read_csv(os.path.join(hf_data_dir, csv))
    
    if "ssupaug" in csv:
        recording_dataset = ssupaug_dataset_handling(recording_dataset, csv_recording, verbose)
    
    add_box_columns_for_norm_only = False
    if include_columns != "all":
        # print(f"include_columns: {include_columns}")
        for col in include_columns:
            assert(col in recording_dataset.columns), f"Requested column {col} not found in csv {csv} - {csv_recording}"
            
        # Handle case where bounding box is not in the include columns
        if "xmin" not in include_columns or "xmax" not in include_columns or "ymin" not in include_columns or "ymax" not in include_columns:
            # if one of them is missing assert all of them are 
            assert("xmin" not in include_columns), "If one of xmin, xmax, ymin, ymax is not in include columns, then none of them must be"
            assert("xmax" not in include_columns), "If one of xmin, xmax, ymin, ymax is not in include columns, then none of them must be"
            assert("ymin" not in include_columns), "If one of xmin, xmax, ymin, ymax is not in include columns, then none of them must be"
            assert("ymax" not in include_columns), "If one of xmin, xmax, ymin, ymax is not in include columns, then none of them must be"
            add_box_columns_for_norm_only = True
            include_columns.extend(["xmin", "xmax", "ymin", "ymax"])
        
        vitpose_col_to_rename = []
        vitpose_col_to_duplicate = []
        for kpname in VITPOSE_KEYPOINTS_NAMES:
            col = "vitpose_" + kpname + "_score"
            if col not in include_columns:
                # this column is not in the include columns, so we need to rename it to avoid adding it in the data
                vitpose_col_to_rename.append(col)
                include_columns.append(col)
            else:
                # this column is already in the include columns, so we need to duplicate it to keep it in the data
                vitpose_col_to_duplicate.append(col)
        
        recording_dataset = recording_dataset[include_columns]
        
        # Create xmin_meta, xmax_meta, ymin_meta, ymax_meta columns for normalization only if needed
        if add_box_columns_for_norm_only:
            # rename xmin, xmax, ymin, ymax to xmin_meta, xmax_meta, ymin_meta, ymax_meta
            recording_dataset = recording_dataset.rename(columns={"xmin": "xmin_meta", "xmax": "xmax_meta", "ymin": "ymin_meta", "ymax": "ymax_meta"})
        else:
            # copy xmin, xmax, ymin, ymax to xmin_meta, xmax_meta, ymin_meta, ymax_meta
            recording_dataset["xmin_meta"] = recording_dataset["xmin"]
            recording_dataset["xmax_meta"] = recording_dataset["xmax"]
            recording_dataset["ymin_meta"] = recording_dataset["ymin"]
            recording_dataset["ymax_meta"] = recording_dataset["ymax"]
            
        for col in vitpose_col_to_rename:
            recording_dataset = recording_dataset.rename(columns={col: col+"_meta"})
            # print(f"[{csv_recording}] renamed {col} to {col}_meta")
        for col in vitpose_col_to_duplicate:
            recording_dataset[col+"_meta"] = recording_dataset[col]
            # print(f"[{csv_recording}] duplicated {col} to {col}_meta")
            
    elif include_columns == "all":
        # duplicate xmin, xmax, ymin, ymax to xmin_meta, xmax_meta, ymin_meta, ymax_meta
        recording_dataset["xmin_meta"] = recording_dataset["xmin"]
        recording_dataset["xmax_meta"] = recording_dataset["xmax"]
        recording_dataset["ymin_meta"] = recording_dataset["ymin"]
        recording_dataset["ymax_meta"] = recording_dataset["ymax"]
        
        for kpname in VITPOSE_KEYPOINTS_NAMES:
            col = "vitpose_" + kpname + "_score"
            # duplicate vitpose_score columns to vitpose_score_meta columns
            recording_dataset[col+"_meta"] = recording_dataset[col]
        
    # Remove mask if not requested to save space
    if not return_masks and "mask_rle" in recording_dataset.columns:
        recording_dataset = recording_dataset.drop(columns=["mask_rle"])
    tac = time.time()
    # prTimer(f"csv dataset {csv} - {csv_recording}", tic, tac)
        
    return recording_dataset, csv_recording


def check_additional_filtering(additional_filtering_dict, track_data: pd.DataFrame) -> np.ndarray[bool]:
    """ Check if the track data is valid according to the additional filtering.
    For now we only support additional filtering as min and max on an existing column.

    Args:
        additional_filtering_dict (dict): additional filtering dictionary
        track_data (pd.DataFrame): track data (extract of the dataset corresponding to a unique track identifier)

    Returns:
        np.ndarray[bool]: mask with length len(track_data)
    """
    
    global_mask = np.ones(len(track_data), dtype=bool)
    for key in additional_filtering_dict:
        if key in track_data.columns:
            key_series = np.array(track_data[key]) # to numpy
            min_mask = np.ones(len(track_data), dtype=bool)
            max_mask = np.ones(len(track_data), dtype=bool)
            if additional_filtering_dict[key]["min"] is not None:
                min_mask = key_series >= additional_filtering_dict[key]["min"]
            if additional_filtering_dict[key]["max"] is not None:
                max_mask = key_series <= additional_filtering_dict[key]["max"]
            
            global_mask[min_mask == False] = False
            global_mask[max_mask == False] = False

    return global_mask

def get_keypoints_mask(min_keypoints_filter, track_data: pd.DataFrame, score_threshold: float = 0.5) -> np.ndarray[bool]:
    """ Check if the track data is valid according to the keypoints filtering (number of ViTPose keypoints with score >= score_threshold).

    Args:
        min_keypoints_filter (int): minimum number of keypoints with score >= score_threshold
        track_data (pd.DataFrame): track data (extract of the dataset corresponding to a unique track identifier)
        score_threshold (float, optional): score threshold. Defaults to 0.5.

    Returns:
        np.ndarray[bool]: mask with length len(track_data)
    """
    keypoints_scores = track_data[VITPOSE_KEYPOINTS_SCORES_COLUMNS]
    keypoints_scores_valid = keypoints_scores >= score_threshold
    keypoints_scores_sum = keypoints_scores_valid.sum(axis=1)
    return np.array(keypoints_scores_sum >= min_keypoints_filter)

def get_autovalidity_mask(track_data: pd.DataFrame) -> np.ndarray[bool]:
    """ Check if the track data is valid according to the autovalidity filtering (validity == "valid") established at dataset creation/curation time.

    Args:
        track_data (pd.DataFrame): track data (extract of the dataset corresponding to a unique track identifier)

    Returns:
        np.ndarray[bool]: mask with length len(track_data)
    """
    return np.array(track_data["validity"] == "valid")

def get_existence_mask(track_data: pd.DataFrame) -> tuple[np.ndarray[bool], np.ndarray[int], np.ndarray[int]]:
    """ Check if the track data exists at each step between the first and the last frame of appearance in the recording, this is to identify missing frames and thus remove or split non continuous tracks.

    Args:
        track_data (pd.DataFrame): track data (extract of the dataset corresponding to a unique track identifier)

    Returns:
        np.ndarray[bool]: existence_mask, shape (max_frame_index - min_frame_index + 1,) boolean existence mask indicating if the track data exists at each step
        np.ndarray[int]: existence_mask_index_to_track_data_index, shape (max_frame_index - min_frame_index + 1,) index in existence mask to index in track data, -1 means it does not exist in track_data
        np.ndarray[int]: track_data_index_to_existence_mask_index, shape (len(track_data),) index in track_data to index in existence mask (i.e. referenced between min and max frame index for this track)
    """
    min_frame_index = track_data["image_index"].min()
    max_frame_index = track_data["image_index"].max()
    existence_mask = np.zeros(max_frame_index - min_frame_index + 1, dtype=bool)
    existence_mask_index_to_track_data_index = np.full(max_frame_index - min_frame_index + 1, -1, dtype=int) # index in array to index in track_data
    track_data_index_to_existence_mask_index = np.zeros(len(track_data), dtype=int) # index in track_data to index in array
    for iter, (_, row) in enumerate(track_data.iterrows()):
        existence_mask[row["image_index"] - min_frame_index] = True
        
        track_data_index_to_existence_mask_index[iter] = row["image_index"] - min_frame_index
        existence_mask_index_to_track_data_index[row["image_index"] - min_frame_index] = iter
        
    return existence_mask, existence_mask_index_to_track_data_index, track_data_index_to_existence_mask_index

def get_first_interaction_index(track_data: pd.DataFrame) -> int:
    # index among the track data
    for iter, (index, row) in enumerate(track_data.iterrows()):
        if row["engagement"] > 0:
            return iter
    return -1 

def get_biggest_mask_index(track_data: pd.DataFrame) -> int:
    """ Get the index of the biggest mask size in the track data.
    """
    masks_size_series = np.array(track_data["mask_size"])
    return np.argmax(masks_size_series)
        
def get_track_input_possible_indices(track_data: pd.DataFrame, 
                                     unique_track_identifier: str, 
                                     input_length_in_frames: int, 
                                     fixed_input_length: bool, 
                                     min_length_in_frames: int, 
                                     max_length_in_frames: int, 
                                     interaction_cutoff: int,
                                     positive_cutoff: int,
                                     force_positive_samples: bool,
                                     ignore_negative_tracks_after_biggest_mask_size: bool,
                                     force_aligment_with_biggest_mask_size: bool,
                                     additional_filtering_dict: dict, 
                                     min_keypoints_filter: int) -> tuple[int, list[tuple[int, int, int]]]:
    """ Get the possible input indices for a track

    Args:
        track_data (pd.DataFrame): track data (extract of the dataset corresponding to a unique track identifier)
        unique_track_identifier (str): unique track identifier
        input_length_in_frames (int): input length in frames
        fixed_input_length (bool): if True, the input length is fixed
        min_length_in_frames (int): minimum length in frames
        max_length_in_frames (int): maximum length in frames
        interaction_cutoff (int): interaction cutoff (corresponds to T_CUT)
        positive_cutoff (int): positive cutoff (corresponds to T_INT)
        force_positive_samples (bool): if True, force positive samples
        ignore_negative_tracks_after_biggest_mask_size (bool): if True, align negative tracks to the biggest mask size
        force_aligment_with_biggest_mask_size (bool): if True, force alignment with the biggest mask size
        additional_filtering_dict (dict): additional filtering dictionary
        min_keypoints_filter (int): minimum number of keypoints with score >= score_threshold

    Returns:
        tuple[int, list[tuple[int, int, int]]]: first interaction index, list of tuples (possible_start_frame, number_of_frames, label)
    """
    
    has_vitpose_keypoints_scores = all([col in track_data.columns for col in VITPOSE_KEYPOINTS_SCORES_COLUMNS])

    min_frames = input_length_in_frames if fixed_input_length else min_length_in_frames
    assert(type(min_frames) == int), "min_length_in_frames must be an integer"
        
    if len(track_data) < min_frames:
        return None, []
    
    additional_filtering_mask = check_additional_filtering(additional_filtering_dict, track_data) # mask with length len(track_data)
    if has_vitpose_keypoints_scores:
        keypoints_mask = get_keypoints_mask(min_keypoints_filter, track_data) # mask with length len(track_data)
    else:
        keypoints_mask = np.ones(len(track_data), dtype=bool)
    autovalidity_mask = get_autovalidity_mask(track_data) # mask with length len(track_data)
    
    existence_mask, existence_mask_index_to_track_data_index, track_data_index_to_existence_mask_index = get_existence_mask(track_data) # existence mask as length max_frame_index - min_frame_index + 1
    
    full_validity_mask = np.zeros((4, existence_mask.shape[0]), dtype=bool) # mask with length max_frame_index - min_frame_index + 1, 4 channels of mask : additional, keypoints, autovalidity, existence
    
    full_validity_mask[0, :][track_data_index_to_existence_mask_index] = additional_filtering_mask
    full_validity_mask[1, :][track_data_index_to_existence_mask_index] = keypoints_mask
    full_validity_mask[2, :][track_data_index_to_existence_mask_index] = autovalidity_mask
    full_validity_mask[3] = existence_mask
            
    full_validity_mask = np.logical_and.reduce(full_validity_mask, axis=0) # shape (max_frame_index - min_frame_index + 1,) if one of the channels is False, the track is not valid at this time
    
    first_interaction_index = get_first_interaction_index(track_data) # index among track_data
    if first_interaction_index != -1:
        interaction_index_in_existence_mask = track_data_index_to_existence_mask_index[first_interaction_index]
        full_validity_mask[interaction_index_in_existence_mask+1:] = False # allow to go up to the start of the interaction
        min_frames_in_segment = min_frames + interaction_cutoff # we need to cut even before the first interaction
        biggest_mask_index_in_existence_mask = None
    else:
        if ignore_negative_tracks_after_biggest_mask_size or force_aligment_with_biggest_mask_size:
            # only consider the part before the biggest mask size
            biggest_mask_index = get_biggest_mask_index(track_data)
            biggest_mask_index_in_existence_mask = track_data_index_to_existence_mask_index[biggest_mask_index]
            full_validity_mask[biggest_mask_index_in_existence_mask+1:] = False
            if force_aligment_with_biggest_mask_size:  
                min_frames_in_segment = min_frames + interaction_cutoff # we will align to interaction_cutoff frames before the interaction
            else:
                min_frames_in_segment = min_frames # we will not align to interaction_cutoff frames before the interaction
        else:
            biggest_mask_index_in_existence_mask = None
            min_frames_in_segment = min_frames            
        interaction_index_in_existence_mask = None


    if full_validity_mask.sum() == 0:
        return None, []
    
    # prDebug(f"{unique_track_identifier} - full_validity_mask shape: valid_track : {full_validity_mask.sum()} / {full_validity_mask.shape[0]} | Interaction index : {interaction_index_in_existence_mask}")
    
    contiguous_true_segments = get_contiguous_true_segments(full_validity_mask) # list of tuples (start, length) with the positions of the true segments and their lengths. Inclusive of the first element. Positions in image indexes relatively to the first image index of the track
    possible_indices = []
    for (start, length) in contiguous_true_segments:
        if length >= min_frames_in_segment:
            start_in_track_data_index = existence_mask_index_to_track_data_index[start]
            assert(start_in_track_data_index != -1), f"Start in track data index is -1 for track {unique_track_identifier} at index {start} it should not happen"
            
            if fixed_input_length:
                possible_starting_points = range(start_in_track_data_index, start_in_track_data_index + length + 1 - min_frames_in_segment) # such that the end of the input is still in the valid segment. 
                # Example with start_in_track_data_index = 100, input_length_in_frames = min_frames = 30, interaction_cutoff = 5, length = 50
                # If not interaction we have range(100, 100+50-30) = range(100, 120)
                # If interaction we have range(100, 100+50-35) = range(100, 115) such that the end of the input of length (min_frames = input_length_in_frames) is at least interaction_cutoff frames before the interaction
                for possible_starting_point in possible_starting_points:
                    label = 0 # no interaction
                    if first_interaction_index != -1:
                        if possible_starting_point + input_length_in_frames >= first_interaction_index - positive_cutoff + 1:
                            label = 1 # interaction if the the input ends at least positive_cutoff frames before the interaction
                        else:
                            label = 0
                            if force_positive_samples:
                                # dont allow for negative samples when possible to force positive samples
                                continue
                    else:
                        if force_aligment_with_biggest_mask_size and possible_starting_point + input_length_in_frames != biggest_mask_index_in_existence_mask - interaction_cutoff + 1:
                            # force to end right on the biggest mask size
                            continue
                        label = 0
                        
                    possible_indices.append((possible_starting_point, input_length_in_frames, label))
                    
            else:
                possible_starting_points = range(start_in_track_data_index, start_in_track_data_index + length - min_frames_in_segment) # such that the end of the input is still in the valid segment
                for seg_count, possible_starting_point in enumerate(possible_starting_points):
                    # Example with start_in_track_data_index = 100, min_length_in_frames = min_frames = 20, max_length_in_frames = 40, interaction_cutoff = 5, length = 50
                    # If not interaction we have possible_starting_points = range(100, 100+50-20+1) = range(100, 130)
                    # Example with seg_count = 5, possible_starting_point = 105
                    max_frames_for_this_starting_point = min(max_length_in_frames, length - seg_count) # Example with seg_count = 5, possible_starting_point = 105 --> min(40, 50-5) = 40 | Example with seg_count = 17, possible_starting_point = 117 --> min(40, 50-17) = 33
                    for possible_frame_count in range(min_length_in_frames, max_frames_for_this_starting_point + 1):
                        if first_interaction_index != -1:
                            if possible_starting_point + possible_frame_count >= first_interaction_index - positive_cutoff + 1:
                                label = 1 # interaction if the the input ends at least positive_cutoff frames before the interaction
                            else:
                                label = 0
                                if force_positive_samples:
                                    # dont allow for negative samples when possible to force positive samples
                                    continue
                        else:
                            # negative sample anyway
                            label = 0
                            
                        possible_indices.append((possible_starting_point, possible_frame_count, label))
    
    return first_interaction_index, possible_indices


def process_track_input(args: dict) -> tuple[str, int, list[tuple[int, int, int]]]:
    
    unique_track_identifier = args["unique_track_identifier"]
    track_data = args["track_data"]
    input_length_in_frames = args["input_length_in_frames"]
    fixed_input_length = args["fixed_input_length"]
    min_length_in_frames = args["min_length_in_frames"]
    max_length_in_frames = args["max_length_in_frames"]
    interaction_cutoff = args["interaction_cutoff"]
    positive_cutoff = args["positive_cutoff"]
    force_positive_samples = args["force_positive_samples"]
    additional_filtering_dict = args["additional_filtering_dict"]
    min_keypoints_filter = args["min_keypoints_filter"]
    ignore_negative_tracks_after_biggest_mask_size = args["ignore_negative_tracks_after_biggest_mask_size"]
    force_aligment_with_biggest_mask_size = args["force_aligment_with_biggest_mask_size"]
    
    first_interaction_index, possible_indices = get_track_input_possible_indices(track_data, 
                                                                                 unique_track_identifier, 
                                                                                 input_length_in_frames, 
                                                                                 fixed_input_length, 
                                                                                 min_length_in_frames, 
                                                                                 max_length_in_frames, 
                                                                                 interaction_cutoff, 
                                                                                 positive_cutoff, 
                                                                                 force_positive_samples, 
                                                                                 ignore_negative_tracks_after_biggest_mask_size,
                                                                                 force_aligment_with_biggest_mask_size,
                                                                                 additional_filtering_dict, 
                                                                                 min_keypoints_filter)
    return (unique_track_identifier, first_interaction_index, possible_indices)
class HUIInteract360(Dataset):
    
    def __init__(self,
                 include_recordings: Union[list, str] = "all",
                 include_tracks: Union[list, str] = "all",
                 include_columns: Union[list, str] = "all",
                 positive_cutoff: int = 60, # in frames
                 interaction_cutoff: int = 5, # in frames
                 fixed_input_length: bool = True,
                 input_length_in_frames: Optional[int] = 10, # if fixed_input_length is True then it should be None
                 min_length_in_frames: Optional[int] = None,
                 max_length_in_frames: Optional[int] = None,
                 subsample_frames: Optional[int] = 1,
                 min_keypoints_filter: int = 9,
                 additional_filtering_dict: dict = {"mask_size": {"min": 1000, "max": 1e7}},
                 return_images: bool = False,
                 return_masks: bool = False,
                 normalize_in_image: bool = True,
                 normalize_keypoints_in_box: bool = False,
                 standardize_data: str = "all",
                 metadata_columns: list = METADATA_COLUMNS,
                 fix_index_per_track: bool = False,
                 fix_index_per_track_list: list = None,
                 force_positive_samples: bool = False,
                 ignore_negative_tracks_after_biggest_mask_size: bool = False,
                 force_aligment_with_biggest_mask_size: bool = False,
                 verbose: bool = True,
                 dataset_revision: str = "3c8a342548534b6b92d32b0099e266962facdf45",
                 ):
        
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
        
        if force_aligment_with_biggest_mask_size:
            assert(ignore_negative_tracks_after_biggest_mask_size), "force_aligment_with_biggest_mask_size requires ignore_negative_tracks_after_biggest_mask_size to be True"
        
        if return_images or return_masks:
            raise NotImplementedError("Not implemented to return images or masks")
        
        if include_columns != "all":
            assert("recording" in include_columns), "recording column must be in include_columns"
            assert("unique_track_identifier" in include_columns), "unique_track_identifier column must be in include_columns"
            assert("image_index" in include_columns), "image_index column must be in include_columns"
        
        if normalize_keypoints_in_box:
            assert(normalize_in_image), "normalize_keypoints_in_box requires normalize_in_image to be True"
            
        if fix_index_per_track_list is not None:
            assert(fix_index_per_track), "fix_index_per_track_list requires fix_index_per_track to be True"
        
        # Set variables
        self.include_recordings = include_recordings
        self.include_tracks = include_tracks
        self.include_columns = include_columns
        self.positive_cutoff = positive_cutoff
        self.interaction_cutoff = interaction_cutoff
        self.fixed_input_length = fixed_input_length
        self.input_length_in_frames = input_length_in_frames
        self.min_length_in_frames = min_length_in_frames
        self.max_length_in_frames = max_length_in_frames
        self.subsample_frames = subsample_frames
        self.min_keypoints_filter = min_keypoints_filter
        self.additional_filtering_dict = additional_filtering_dict
        self.force_positive_samples = force_positive_samples
        self.ignore_negative_tracks_after_biggest_mask_size = ignore_negative_tracks_after_biggest_mask_size
        self.force_aligment_with_biggest_mask_size = force_aligment_with_biggest_mask_size
        self.return_images = return_images
        self.return_masks = return_masks
        self.normalize_in_image = normalize_in_image
        self.normalize_keypoints_in_box = normalize_keypoints_in_box
        self.standardize_data = standardize_data
        self.metadata_columns = metadata_columns + ["xmin_meta", "xmax_meta", "ymin_meta", "ymax_meta"] + VITPOSE_KEYPOINTS_SCORES_COLUMNS # append bounding box columns used for normalization (even in the absence of bounding box in the actual data input), created at loading time
        self.verbose = verbose
        self.fix_index_per_track = fix_index_per_track
    
        # Download the dataset
        hf_local_dir = os.path.join(here, "..", "datasets", "hf_data")
        hf_data_dir = snapshot_download(repo_id="rlorlou/HUI360", repo_type="dataset", ignore_patterns="ignore.json", revision = dataset_revision, local_dir = hf_local_dir, max_workers=8, dry_run=False)        
        csvs_in_data_dir = [f for f in os.listdir(hf_data_dir) if f.endswith(".csv")]
        csvs_in_data_dir.sort()
        
        prInfo(f"Got {len(csvs_in_data_dir)} csvs in hf_data directory")
        
        # Load the dataset, recording by recording with some filtering using multiprocessing
        keep_recordings = []
        existing_recordings = []

        # Prepare arg list to avoid closure
        arg_list = []
        for csv in csvs_in_data_dir:
            arg_list.append((csv, hf_data_dir, include_recordings, include_columns, return_masks, self.verbose))

        results = []
        mp_ctx = mp.get_context("fork" if hasattr(os, 'fork') else "spawn")
        with mp_ctx.Pool(processes=min(int(mp.cpu_count()/2), len(arg_list))) as pool:
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
        if include_tracks != "all":
            unique_tracks_before = self.dataset["unique_track_identifier"].unique() 
            self.dataset = self.dataset[self.dataset["unique_track_identifier"].isin(include_tracks)]
            unique_tracks_after = self.dataset["unique_track_identifier"].unique()
            self.log(f"Filtered by tracks and kept {len(unique_tracks_after)} tracks out of {len(unique_tracks_before)} (requested {len(include_tracks)} tracks)", "success")
            if len(unique_tracks_after) != len(include_tracks):
                self.log(f"Requested {len(include_tracks)} tracks but got {len(unique_tracks_after)} tracks", "warning")
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
                                             "ignore_negative_tracks_after_biggest_mask_size": self.ignore_negative_tracks_after_biggest_mask_size,
                                             "force_aligment_with_biggest_mask_size": self.force_aligment_with_biggest_mask_size,
                                             "additional_filtering_dict": self.additional_filtering_dict, 
                                             "min_keypoints_filter": self.min_keypoints_filter})
            
        self.unique_track_to_first_interaction_index = {}

        # Use torch multiprocessing for parallelization
        self.log(f"Running with torch multiprocessing for parallelization of {len(unique_tracks_identifiers)} tracks (process input and find possible starting points)", "info")
        with mp_ctx.Pool(processes=min(int(mp.cpu_count()/2), len(unique_tracks_identifiers))) as pool:
            track_results = pool.map(process_track_input, process_track_input_args)

        self.index_choice_by_unique_track_identifier = {}
        for unique_track_iter, (unique_track, first_interaction_index, possible_indices) in enumerate(track_results):
            self.unique_track_to_first_interaction_index[unique_track] = first_interaction_index
            
            if len(possible_indices) > 0:
                self.input_by_tracks[unique_track] = possible_indices
                if self.fix_index_per_track:
                    # fix it for each track for the every call to __getitem__
                    if fix_index_per_track_list is not None:
                        # to fix it even more from a run to another
                        self.index_choice_by_unique_track_identifier[unique_track] = fix_index_per_track_list[unique_track_iter%len(fix_index_per_track_list)]%len(possible_indices)
                    else:
                        self.index_choice_by_unique_track_identifier[unique_track] = torch.randint(0, len(possible_indices), (1,)).item()

                else:
                    self.index_choice_by_unique_track_identifier[unique_track] = None
                
        self.idx_to_track = list(self.input_by_tracks.keys()) # to query a track when using __getitem__
            
        # Do some counting
        total_possible_inputs = 0
        total_possible_inputs_with_interaction = 0
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
                
            total_possible_inputs += total_possible_inputs_for_this_track
            total_possible_inputs_with_interaction += total_possible_inputs_with_interaction_for_this_track

        self.total_inputs = total_possible_inputs
        
        self.total_positives = total_tracks_with_possible_interaction
        self.total_negatives = total_tracks - total_tracks_with_possible_interaction
        self.total_possible_positives = total_possible_inputs_with_interaction # possible positives !
        self.total_possible_negatives = total_possible_inputs - total_possible_inputs_with_interaction # possible negatives !

        #### Write to a file the list of tracks with valid input ####
        # timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        # save_tracks_path = os.path.join(here, "..", "utils", "tests", f"{timestamp}_tracks_with_valid_input_len_{self.input_length_in_frames}_int_{self.interaction_cutoff}_pos_{self.positive_cutoff}_force_pos_{self.force_positive_samples}_ignore_neg_{self.ignore_negative_tracks_after_biggest_mask_size}_force_align_{self.force_aligment_with_biggest_mask_size}.txt")
        # with open(save_tracks_path, "w") as f:
        #     for track in tracks_with_valid_input:
        #         f.write(f"{track}\n")
        # prDebug(f"Saved tracks with valid input to file {save_tracks_path}")
        
        # save_track_with_possible_interaction_path = os.path.join(here, "..", "utils", "tests", f"{timestamp}_tracks_with_possible_interaction_len_{self.input_length_in_frames}_int_{self.interaction_cutoff}_pos_{self.positive_cutoff}_force_pos_{self.force_positive_samples}_ignore_neg_{self.ignore_negative_tracks_after_biggest_mask_size}_force_align_{self.force_aligment_with_biggest_mask_size}.txt")
        # with open(save_track_with_possible_interaction_path, "w") as f:
        #     for track in tracks_with_possible_interaction:
        #         f.write(f"{track}\n")
        # prDebug(f"Saved tracks with possible interaction to file {save_track_with_possible_interaction_path}")
        
        #######################################################################

        tac = time.time()
        
        # prTimer(f"Total possible inputs computation", tic, tac)
        self.log(f"Total possible inputs: {total_possible_inputs} among {len(self.input_by_tracks)} tracks", "success")
        self.log(f"Total possible inputs with interaction: {total_possible_inputs_with_interaction} among {len(tracks_with_possible_interaction)} tracks", "success")
        
        self.log(f"Total tracks: {total_tracks}", "success")
        self.log(f"Total tracks with possible interaction: {total_tracks_with_possible_interaction}", "success")
    
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
        
        for col in input_df.columns:
            if col.startswith("vitpose_") or col.startswith("sapiens_"):
                if self.normalize_keypoints_in_box:
                    if col.endswith("_x"):
                        input_df[col] = (input_df[col] - normalize_bbox_df["xmin_meta"]) / (normalize_bbox_df["xmax_meta"] - normalize_bbox_df["xmin_meta"])
                    elif col.endswith("_y"):
                        input_df[col] = (input_df[col] - normalize_bbox_df["ymin_meta"]) / (normalize_bbox_df["ymax_meta"] - normalize_bbox_df["ymin_meta"])
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
                    
                elif (col.startswith("vitpose_") or col.startswith("sapiens_")) and self.normalize_keypoints_in_box:
                    # we womm use _box_norm for keypoints in box coordinates
                    if col.endswith("_x"):
                        input_df[col] = (input_df[col] - HUI_NORMALIZATION_VALUES[col+"_box_norm"]["mean"]) / HUI_NORMALIZATION_VALUES[col+"_box_norm"]["std"]
                    elif col.endswith("_y"):
                        input_df[col] = (input_df[col] - HUI_NORMALIZATION_VALUES[col+"_box_norm"]["mean"]) / HUI_NORMALIZATION_VALUES[col+"_box_norm"]["std"]


                elif (col.startswith("vitpose_") or col.startswith("sapiens_")) and not self.normalize_keypoints_in_box:
                    # we womm use _image_norm for keypoints in image coordinates
                    if col.endswith("_x"):
                        input_df[col] = (input_df[col] - HUI_NORMALIZATION_VALUES[col+"_image_norm"]["mean"]) / HUI_NORMALIZATION_VALUES[col+"_image_norm"]["std"]
                    elif col.endswith("_y"):
                        input_df[col] = (input_df[col] - HUI_NORMALIZATION_VALUES[col+"_image_norm"]["mean"]) / HUI_NORMALIZATION_VALUES[col+"_image_norm"]["std"]
                        
        elif self.standardize_data == "mask_only":
            if "mask_size" in input_df.columns:
                input_df["mask_size"] = (input_df["mask_size"] - HUI_NORMALIZATION_VALUES["mask_size"]["mean"]) / HUI_NORMALIZATION_VALUES["mask_size"]["std"]
                
        return input_df
        
    def __len__(self):
        return len(self.idx_to_track)

    def __getitem__(self, idx):
        
        tic = time.time()
        unique_track_identifier = self.idx_to_track[idx]
        possible_inputs = self.input_by_tracks[unique_track_identifier]

        track_data = self.datasets_by_unique_track_identifier[unique_track_identifier]
        first_interaction_index = self.unique_track_to_first_interaction_index[unique_track_identifier]
        # track_data = track_data.sort_values(by="image_index") # already sorted by image_index
        
        if self.fix_index_per_track:
            random_index = self.index_choice_by_unique_track_identifier[unique_track_identifier]
        else:
            random_index = torch.randint(0, len(possible_inputs), (1,)).item()
            
        input_index, number_of_frames, label = possible_inputs[random_index]

        image_indexes_track = range(input_index, input_index + number_of_frames, self.subsample_frames)
        # input_df = track_data.iloc[input_index:input_index + number_of_frames]
        # input_df = input_df.iloc[::self.subsample_frames] # apply stride to the input dataframe
        input_df = track_data.iloc[image_indexes_track]

        metadata_df = input_df[self.metadata_columns_in_dataset]
        metadata_dict = {}
        metadata_dict["unique_track_identifier"] = metadata_df["unique_track_identifier"].iloc[0]
        metadata_dict["image_size"] = (int(metadata_df["image_width"].iloc[0]), int(metadata_df["image_height"].iloc[0]))
        metadata_dict["image_indexes"] = [int(metadata_df["image_index"].iloc[i]) for i in range(len(metadata_df))]
        metadata_dict["image_indexes_track"] = [i for i in image_indexes_track]
        metadata_dict["engagements"] = [int(metadata_df["engagement"].iloc[i]) for i in range(len(metadata_df))]
        metadata_dict["time_to_first_interaction"] = metadata_df["time_to_first_interaction"].iloc[-1]
        metadata_dict["first_interaction_index"] = first_interaction_index
        metadata_dict["index_choice"] = random_index
        metadata_dict["input_index"] = input_index
        metadata_dict["number_of_frames"] = number_of_frames
        metadata_dict["xmin_meta"] = metadata_df["xmin_meta"].values
        metadata_dict["xmax_meta"] = metadata_df["xmax_meta"].values
        metadata_dict["ymin_meta"] = metadata_df["ymin_meta"].values
        metadata_dict["ymax_meta"] = metadata_df["ymax_meta"].values
        metadata_dict["time_to_interaction_by_frame"] = metadata_df["time_to_first_interaction"].values
        
        image_height = metadata_df["image_height"].iloc[0]
        image_width = metadata_df["image_width"].iloc[0]
        
        input_df = input_df[self.data_columns_in_dataset]
        if self.normalize_in_image: # and "xmin" in self.data_columns_in_dataset and "ymin" in self.data_columns_in_dataset and "xmax" in self.data_columns_in_dataset and "ymax" in self.data_columns_in_dataset:
            normalize_bbox_df = metadata_df[["xmin_meta", "xmax_meta", "ymin_meta", "ymax_meta"]]
            input_df = self.normalize_data(input_df, normalize_bbox_df, image_height, image_width)
        
        # convert all columns to float32
        input_df = input_df.astype(np.float32)

        input_tensor = torch.tensor(input_df.values)
        label = torch.tensor(label)
        # prDebug(f"Input data: {input_tensor.shape}, label: {label}, metadata: {metadata_dict}")
        
        tac = time.time()
        if self.verbose:
            prTimer(f"__getitem__ for {unique_track_identifier}", tic, tac)
        return input_tensor, label, metadata_dict