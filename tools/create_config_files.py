
import itertools
import sys
import os
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, ".."))
from utils.print_utils import *
from utils.other_utils import *
from utils.data_utils import SAPIENS_FACE_KEYPOINTS_LIST
from utils.sapiens_selected import SAPIENS_EXCLUDING_FACE_KEYPOINTS_LIST
from copy import copy
import shutil
import socket
hostname = socket.gethostname()

FIX_LIST = [3662, 5427, 3606, 3726, 3417, 6031, 7527, 1501, 4501, 9588, 2712, 4509, 2752, 57, 9256, 3417, 8694, 9336, 6870, 3587, 2675, 3613, 9281, 4883, 7570, 8967, 1654, 5194, 9746, 4310, 2848, 9954]
COLUMNS_MASK_SIZE_ONLY = [
            "recording", "episode", "image_height", "image_width", "unique_track_identifier", 
            "track_id", "image_file", "image_index", "validity", "current_segment", 
            "total_segments", "position_in_segment", "length_of_current_segment", 
            "timestamp", "timestamp_sec", "timestamp_track", "engagement", 
            "time_to_first_interaction", "mask_size"
        ]


COLUMNS_MASK_SIZE_BOX = [
            "recording", "episode", "image_height", "image_width", "unique_track_identifier", 
            "track_id", "image_file", "image_index", "validity", "current_segment", 
            "total_segments", "position_in_segment", "length_of_current_segment", 
            "timestamp", "timestamp_sec", "timestamp_track", "engagement", 
            "time_to_first_interaction", "mask_size", "xmin", "ymin", 
            "xmax", "ymax"
        ]

COLUMNS_MASK_SIZE_BOX_SHOULDERS_EARS_EYES = [
            "recording", "episode", "image_height", "image_width", "unique_track_identifier", 
            "track_id", "image_file", "image_index", "validity", "current_segment", 
            "total_segments", "position_in_segment", "length_of_current_segment", 
            "timestamp", "timestamp_sec", "timestamp_track", "engagement", 
            "time_to_first_interaction", "mask_size", "xmin", "ymin", 
            "xmax", "ymax", 
            "vitpose_left_shoulder_x", "vitpose_left_shoulder_y", "vitpose_left_shoulder_score", 
            "vitpose_right_shoulder_x", "vitpose_right_shoulder_y", "vitpose_right_shoulder_score", 
            "vitpose_left_ear_x", "vitpose_left_ear_y", "vitpose_left_ear_score", 
            "vitpose_right_ear_x", "vitpose_right_ear_y", "vitpose_right_ear_score", 
            "vitpose_left_eye_x", "vitpose_left_eye_y", "vitpose_left_eye_score", 
            "vitpose_right_eye_x", "vitpose_right_eye_y", "vitpose_right_eye_score", 
        ]

COLUMNS_MASK_SIZE_BOX_VITPOSE = [
            "recording", "episode", "image_height", "image_width", "unique_track_identifier", 
            "track_id", "image_file", "image_index", "validity", "current_segment", 
            "total_segments", "position_in_segment", "length_of_current_segment", 
            "timestamp", "timestamp_sec", "timestamp_track", "engagement", 
            "time_to_first_interaction", "mask_size", "xmin", "ymin", 
            "xmax", "ymax", "vitpose_nose_x", "vitpose_nose_y", "vitpose_nose_score", 
            "vitpose_left_eye_x", "vitpose_left_eye_y", "vitpose_left_eye_score", 
            "vitpose_right_eye_x", "vitpose_right_eye_y", "vitpose_right_eye_score", 
            "vitpose_left_ear_x", "vitpose_left_ear_y", "vitpose_left_ear_score", 
            "vitpose_right_ear_x", "vitpose_right_ear_y", "vitpose_right_ear_score", 
            "vitpose_left_shoulder_x", "vitpose_left_shoulder_y", "vitpose_left_shoulder_score", 
            "vitpose_right_shoulder_x", "vitpose_right_shoulder_y", "vitpose_right_shoulder_score", 
            "vitpose_left_elbow_x", "vitpose_left_elbow_y", "vitpose_left_elbow_score", 
            "vitpose_right_elbow_x", "vitpose_right_elbow_y", "vitpose_right_elbow_score", 
            "vitpose_left_wrist_x", "vitpose_left_wrist_y", "vitpose_left_wrist_score", 
            "vitpose_right_wrist_x", "vitpose_right_wrist_y", "vitpose_right_wrist_score", 
            "vitpose_left_hip_x", "vitpose_left_hip_y", "vitpose_left_hip_score", 
            "vitpose_right_hip_x", "vitpose_right_hip_y", "vitpose_right_hip_score", 
            "vitpose_left_knee_x", "vitpose_left_knee_y", "vitpose_left_knee_score", 
            "vitpose_right_knee_x", "vitpose_right_knee_y", "vitpose_right_knee_score", 
            "vitpose_left_ankle_x", "vitpose_left_ankle_y", "vitpose_left_ankle_score", 
            "vitpose_right_ankle_x", "vitpose_right_ankle_y", "vitpose_right_ankle_score"
        ]

COLUMNS_MASK_SIZE_VITPOSE = [
            "recording", "episode", "image_height", "image_width", "unique_track_identifier", 
            "track_id", "image_file", "image_index", "validity", "current_segment", 
            "total_segments", "position_in_segment", "length_of_current_segment", 
            "timestamp", "timestamp_sec", "timestamp_track", "engagement", 
            "time_to_first_interaction", "mask_size", "vitpose_nose_x", "vitpose_nose_y", "vitpose_nose_score", 
            "vitpose_left_eye_x", "vitpose_left_eye_y", "vitpose_left_eye_score", 
            "vitpose_right_eye_x", "vitpose_right_eye_y", "vitpose_right_eye_score", 
            "vitpose_left_ear_x", "vitpose_left_ear_y", "vitpose_left_ear_score", 
            "vitpose_right_ear_x", "vitpose_right_ear_y", "vitpose_right_ear_score", 
            "vitpose_left_shoulder_x", "vitpose_left_shoulder_y", "vitpose_left_shoulder_score", 
            "vitpose_right_shoulder_x", "vitpose_right_shoulder_y", "vitpose_right_shoulder_score", 
            "vitpose_left_elbow_x", "vitpose_left_elbow_y", "vitpose_left_elbow_score", 
            "vitpose_right_elbow_x", "vitpose_right_elbow_y", "vitpose_right_elbow_score", 
            "vitpose_left_wrist_x", "vitpose_left_wrist_y", "vitpose_left_wrist_score", 
            "vitpose_right_wrist_x", "vitpose_right_wrist_y", "vitpose_right_wrist_score", 
            "vitpose_left_hip_x", "vitpose_left_hip_y", "vitpose_left_hip_score", 
            "vitpose_right_hip_x", "vitpose_right_hip_y", "vitpose_right_hip_score", 
            "vitpose_left_knee_x", "vitpose_left_knee_y", "vitpose_left_knee_score", 
            "vitpose_right_knee_x", "vitpose_right_knee_y", "vitpose_right_knee_score", 
            "vitpose_left_ankle_x", "vitpose_left_ankle_y", "vitpose_left_ankle_score", 
            "vitpose_right_ankle_x", "vitpose_right_ankle_y", "vitpose_right_ankle_score"
        ]

FEATURES_SET_D1 = COLUMNS_MASK_SIZE_ONLY
FEATURES_SET_D2 = COLUMNS_MASK_SIZE_BOX
FEATURES_SET_D3 = COLUMNS_MASK_SIZE_BOX_VITPOSE
FEATURES_SET_D4 = COLUMNS_MASK_SIZE_BOX_VITPOSE + SAPIENS_FACE_KEYPOINTS_LIST
FEATURES_SET_D5 = "all"

FEATURES_SET_D6 = COLUMNS_MASK_SIZE_BOX_SHOULDERS_EARS_EYES
FEATURES_SET_D7 = COLUMNS_MASK_SIZE_VITPOSE

FEATURES_SET_D8 = COLUMNS_MASK_SIZE_BOX_VITPOSE + SAPIENS_EXCLUDING_FACE_KEYPOINTS_LIST
FEATURES_SET_D9 = COLUMNS_MASK_SIZE_BOX + SAPIENS_EXCLUDING_FACE_KEYPOINTS_LIST

# RECORDING_TO_SETUP = {"rosbag2_2025_07_07-10_24_20":"EntranceCBack-1","rosbag2_2025_07_07-10_49_31":"Cafeteria-1","rosbag2_2025_07_07-11_16_10":"Room104-1","rosbag2_2025_07_07-12_38_45":"Room104-1","rosbag2_2025_07_07-15_33_32":"Room104-1","rosbag2_2025_07_10-10_29_13":"Room005-1","rosbag2_2025_07_10-15_47_18":"Room005-1","rosbag2_2025_07_11-10_28_07":"MainEntrance-1","rosbag2_2025_07_11-11_18_00":"MainEntrance-1","rosbag2_2025_07_11-13_27_26":"CoffeeB-1","rosbag2_2025_07_11-14_54_55":"CoffeeB-1","rosbag2_2025_07_15-12_39_21":"Room104-2","rosbag2_2025_07_15-13_41_01":"Room104-2","rosbag2_2025_07_15-14_48_22":"Room104-2","rosbag2_2025_07_16-13_12_03":"EntranceCBack-2","rosbag2_2025_07_16-14_07_49":"EntranceCBack-2","rosbag2_2025_07_16-15_50_45":"EntranceCBack-2","rosbag2_2025_07_17-11_28_34":"CoffeeB-2","rosbag2_2025_07_17-12_52_12":"CoffeeB-2","rosbag2_2025_07_18-10_37_07":"MainEntrance-2","rosbag2_2025_07_21-10_22_11":"Room104-3","rosbag2_2025_07_21-11_56_40":"Room104-3","rosbag2_2025_07_21-13_09_22":"Room104-3","rosbag2_2025_07_21-14_11_37":"Room104-3","rosbag2_2025_07_21-15_15_07":"Room104-3","rosbag2_2025_07_22-09_38_18":"Cafeteria-2","rosbag2_2025_07_22-10_59_25":"Cafeteria-2","rosbag2_2025_07_22-12_18_30":"Cafeteria-2","rosbag2_2025_07_22-13_30_39":"Cafeteria-2","rosbag2_2025_07_23-11_01_56":"EntranceCBack-3","rosbag2_2025_07_23-12_18_45":"EntranceCBack-3","rosbag2_2025_07_23-13_17_40":"EntranceCBack-3","rosbag2_2025_07_23-14_30_55":"EntranceCBack-3","rosbag2_2025_07_24-10_41_01":"CoffeeB-3","rosbag2_2025_07_24-12_14_56":"CoffeeB-3","rosbag2_2025_07_24-13_33_54":"CoffeeB-3","rosbag2_2025_07_24-14_33_36":"CoffeeB-3","rosbag2_2025_07_25-10_52_32":"MainEntrance-3","rosbag2_2025_07_25-14_14_16":"MainEntrance-3","rosbag2_2025_07_28-10_18_10":"Room104-4","rosbag2_2025_07_28-11_25_33":"Room104-4","rosbag2_2025_07_28-13_05_46":"Room104-4","rosbag2_2025_07_28-14_19_07":"Room104-4","rosbag2_2025_07_29-10_23_14":"EntranceCBack-4","rosbag2_2025_07_29-13_17_18":"EntranceCBack-4","rosbag2_2025_07_29-14_09_46":"EntranceCBack-4","rosbag2_2025_10_07-15_03_48":"EntranceCFacing-1","rosbag2_2025_10_07-16_21_39":"EntranceCFacing-1","rosbag2_2025_10_09-08_52_20":"MainEntrance-4","rosbag2_2025_10_09-10_23_38":"MainHallway-1","rosbag2_2025_10_09-17_37_23":"EntranceCFacing-2","rosbag2_2025_10_09-18_50_21":"EntranceCFacing-2","rosbag2_2025_10_15-09_37_49":"MainHallway-2","rosbag2_2025_10_15-11_03_05":"MainHallway-2","rosbag2_2025_10_15-12_02_00":"EntranceCBack-5","rosbag2_2025_10_15-12_27_14":"EntranceCFacing-3","rosbag2_2025_10_15-13_11_27":"EntranceCBack-6","rosbag2_2025_10_15-14_02_29":"EntranceCBack-6","rosbag2_2025_10_15-14_30_03":"EntranceCBack-6","rosbag2_2025_10_16-09_27_48":"EntranceCBack-7","rosbag2_2025_10_16-11_29_56":"EntranceCFacing-5","rosbag2_2025_10_16-12_47_57":"EntranceCBack-8","rosbag2_2025_10_17-13_19_29":"Room104-5","rosbag2_2025_10_17-14_28_15":"Room104-5","rosbag2_2025_10_17-15_11_00":"Room104-5","rosbag2_2025_10_17-16_47_09":"Room104-5","rosbag2_2025_10_20-10_13_32":"Bulle12X-1","rosbag2_2025_10_20-11_51_02":"Bulle12X-1","rosbag2_2025_10_20-16_02_46":"Bulle12X-1"}

# RECORDING_TO_PLACE = {"rosbag2_2025_07_07-10_24_20":"EntranceCBack","rosbag2_2025_07_07-10_49_31":"Cafeteria","rosbag2_2025_07_07-11_16_10":"Room104","rosbag2_2025_07_07-12_38_45":"Room104","rosbag2_2025_07_07-15_33_32":"Room104","rosbag2_2025_07_10-10_29_13":"Room005","rosbag2_2025_07_10-15_47_18":"Room005","rosbag2_2025_07_11-10_28_07":"MainEntrance","rosbag2_2025_07_11-11_18_00":"MainEntrance","rosbag2_2025_07_11-13_27_26":"CoffeeB","rosbag2_2025_07_11-14_54_55":"CoffeeB","rosbag2_2025_07_15-12_39_21":"Room104","rosbag2_2025_07_15-13_41_01":"Room104","rosbag2_2025_07_15-14_48_22":"Room104","rosbag2_2025_07_16-13_12_03":"EntranceCBack","rosbag2_2025_07_16-14_07_49":"EntranceCBack","rosbag2_2025_07_16-15_50_45":"EntranceCBack","rosbag2_2025_07_17-11_28_34":"CoffeeB","rosbag2_2025_07_17-12_52_12":"CoffeeB","rosbag2_2025_07_18-10_37_07":"MainEntrance","rosbag2_2025_07_21-10_22_11":"Room104","rosbag2_2025_07_21-11_56_40":"Room104","rosbag2_2025_07_21-13_09_22":"Room104","rosbag2_2025_07_21-14_11_37":"Room104","rosbag2_2025_07_21-15_15_07":"Room104","rosbag2_2025_07_22-09_38_18":"Cafeteria","rosbag2_2025_07_22-10_59_25":"Cafeteria","rosbag2_2025_07_22-12_18_30":"Cafeteria","rosbag2_2025_07_22-13_30_39":"Cafeteria","rosbag2_2025_07_23-11_01_56":"EntranceCBack","rosbag2_2025_07_23-12_18_45":"EntranceCBack","rosbag2_2025_07_23-13_17_40":"EntranceCBack","rosbag2_2025_07_23-14_30_55":"EntranceCBack","rosbag2_2025_07_24-10_41_01":"CoffeeB","rosbag2_2025_07_24-12_14_56":"CoffeeB","rosbag2_2025_07_24-13_33_54":"CoffeeB","rosbag2_2025_07_24-14_33_36":"CoffeeB","rosbag2_2025_07_25-10_52_32":"MainEntrance","rosbag2_2025_07_25-14_14_16":"MainEntrance","rosbag2_2025_07_28-10_18_10":"Room104","rosbag2_2025_07_28-11_25_33":"Room104","rosbag2_2025_07_28-13_05_46":"Room104","rosbag2_2025_07_28-14_19_07":"Room104","rosbag2_2025_07_29-10_23_14":"EntranceCBack","rosbag2_2025_07_29-13_17_18":"EntranceCBack","rosbag2_2025_07_29-14_09_46":"EntranceCBack","rosbag2_2025_10_07-15_03_48":"EntranceCFacing","rosbag2_2025_10_07-16_21_39":"EntranceCFacing","rosbag2_2025_10_09-08_52_20":"MainEntrance","rosbag2_2025_10_09-10_23_38":"MainHallway","rosbag2_2025_10_09-17_37_23":"EntranceCFacing","rosbag2_2025_10_09-18_50_21":"EntranceCFacing","rosbag2_2025_10_15-09_37_49":"MainHallway","rosbag2_2025_10_15-11_03_05":"MainHallway","rosbag2_2025_10_15-12_02_00":"EntranceCBack","rosbag2_2025_10_15-12_27_14":"EntranceCFacing","rosbag2_2025_10_15-13_11_27":"EntranceCBack","rosbag2_2025_10_15-14_02_29":"EntranceCBack","rosbag2_2025_10_15-14_30_03":"EntranceCBack","rosbag2_2025_10_16-09_27_48":"EntranceCBack","rosbag2_2025_10_16-11_29_56":"EntranceCFacing","rosbag2_2025_10_16-12_47_57":"EntranceCBack","rosbag2_2025_10_17-13_19_29":"Room104","rosbag2_2025_10_17-14_28_15":"Room104","rosbag2_2025_10_17-15_11_00":"Room104","rosbag2_2025_10_17-16_47_09":"Room104","rosbag2_2025_10_20-10_13_32":"Bulle12X","rosbag2_2025_10_20-11_51_02":"Bulle12X","rosbag2_2025_10_20-16_02_46":"Bulle12X",}

# UNIQUE_PLACES = ['Room005', 'EntranceCBack', 'Bulle12X', 'MainEntrance', 'CoffeeB', 'Room104', 'MainHallway', 'Cafeteria', 'EntranceCFacing']
# UNIQUE_SETUPS = ['Bulle12X-1', 'Cafeteria-1', 'Cafeteria-2', 'CoffeeB-1', 'CoffeeB-2', 'CoffeeB-3', 'EntranceCBack-1', 'EntranceCBack-2', 'EntranceCBack-3', 'EntranceCBack-4', 'EntranceCBack-5', 'EntranceCBack-6', 'EntranceCBack-7', 'EntranceCBack-8', 'EntranceCFacing-1', 'EntranceCFacing-2', 'EntranceCFacing-3', 'EntranceCFacing-5', 'MainEntrance-1', 'MainEntrance-2', 'MainEntrance-3', 'MainEntrance-4', 'MainHallway-1', 'MainHallway-2', 'Room005-1', 'Room104-1', 'Room104-2', 'Room104-3', 'Room104-4', 'Room104-5']
    

ROOM104_SETUP_TO_RECORDINGS = {'Room104-1': ['rosbag2_2025_07_07-11_16_10', 'rosbag2_2025_07_07-12_38_45', 'rosbag2_2025_07_07-15_33_32'], 
                               'Room104-2': ['rosbag2_2025_07_15-12_39_21', 'rosbag2_2025_07_15-13_41_01', 'rosbag2_2025_07_15-14_48_22'], 
                               'Room104-3': ['rosbag2_2025_07_21-10_22_11', 'rosbag2_2025_07_21-11_56_40', 'rosbag2_2025_07_21-13_09_22', 'rosbag2_2025_07_21-14_11_37', 'rosbag2_2025_07_21-15_15_07'], 
                               'Room104-4': ['rosbag2_2025_07_28-10_18_10', 'rosbag2_2025_07_28-11_25_33', 'rosbag2_2025_07_28-13_05_46', 'rosbag2_2025_07_28-14_19_07'], 
                               'Room104-5': ['rosbag2_2025_10_17-13_19_29', 'rosbag2_2025_10_17-14_28_15', 'rosbag2_2025_10_17-15_11_00', 'rosbag2_2025_10_17-16_47_09']}

ENTRANCECBACK_SETUP_TO_RECORDINGS = {'EntranceCBack-1': ['rosbag2_2025_07_07-10_24_20'], 
                                      'EntranceCBack-2': ['rosbag2_2025_07_16-13_12_03', 'rosbag2_2025_07_16-14_07_49', 'rosbag2_2025_07_16-15_50_45'], 
                                      'EntranceCBack-3': ['rosbag2_2025_07_23-11_01_56', 'rosbag2_2025_07_23-12_18_45', 'rosbag2_2025_07_23-13_17_40', 'rosbag2_2025_07_23-14_30_55'], 
                                      'EntranceCBack-4': ['rosbag2_2025_07_29-10_23_14', 'rosbag2_2025_07_29-13_17_18', 'rosbag2_2025_07_29-14_09_46'], 
                                      'EntranceCBack-5': ['rosbag2_2025_10_15-12_02_00'], 
                                      'EntranceCBack-6': ['rosbag2_2025_10_15-13_11_27', 'rosbag2_2025_10_15-14_02_29', 'rosbag2_2025_10_15-14_30_03'], 
                                      'EntranceCBack-7': ['rosbag2_2025_10_16-09_27_48'], 
                                      'EntranceCBack-8': ['rosbag2_2025_10_16-12_47_57']}

UNIQUE_PLACES_RECORDINGS = {
    "Room005": ['rosbag2_2025_07_10-10_29_13', 'rosbag2_2025_07_10-15_47_18'],
    "EntranceCBack": ['rosbag2_2025_07_07-10_24_20', 'rosbag2_2025_07_16-13_12_03', 'rosbag2_2025_07_16-14_07_49', 'rosbag2_2025_07_16-15_50_45', 'rosbag2_2025_07_23-11_01_56', 'rosbag2_2025_07_23-12_18_45', 'rosbag2_2025_07_23-13_17_40', 'rosbag2_2025_07_23-14_30_55', 'rosbag2_2025_07_29-10_23_14', 'rosbag2_2025_07_29-13_17_18', 'rosbag2_2025_07_29-14_09_46', 'rosbag2_2025_10_15-12_02_00', 'rosbag2_2025_10_15-13_11_27', 'rosbag2_2025_10_15-14_02_29', 'rosbag2_2025_10_15-14_30_03', 'rosbag2_2025_10_16-09_27_48', 'rosbag2_2025_10_16-12_47_57'],
    "Bulle12X": ['rosbag2_2025_10_20-10_13_32', 'rosbag2_2025_10_20-11_51_02', 'rosbag2_2025_10_20-16_02_46'],
    "MainEntrance": ['rosbag2_2025_07_11-10_28_07', 'rosbag2_2025_07_11-11_18_00', 'rosbag2_2025_07_18-10_37_07', 'rosbag2_2025_07_25-10_52_32', 'rosbag2_2025_07_25-14_14_16', 'rosbag2_2025_10_09-08_52_20'],
    "CoffeeB": ['rosbag2_2025_07_11-13_27_26', 'rosbag2_2025_07_11-14_54_55', 'rosbag2_2025_07_17-11_28_34', 'rosbag2_2025_07_17-12_52_12', 'rosbag2_2025_07_24-10_41_01', 'rosbag2_2025_07_24-12_14_56', 'rosbag2_2025_07_24-13_33_54', 'rosbag2_2025_07_24-14_33_36'],
    "Room104": ['rosbag2_2025_07_07-11_16_10', 'rosbag2_2025_07_07-12_38_45', 'rosbag2_2025_07_07-15_33_32', 'rosbag2_2025_07_15-12_39_21', 'rosbag2_2025_07_15-13_41_01', 'rosbag2_2025_07_15-14_48_22', 'rosbag2_2025_07_21-10_22_11', 'rosbag2_2025_07_21-11_56_40', 'rosbag2_2025_07_21-13_09_22', 'rosbag2_2025_07_21-14_11_37', 'rosbag2_2025_07_21-15_15_07', 'rosbag2_2025_07_28-10_18_10', 'rosbag2_2025_07_28-11_25_33', 'rosbag2_2025_07_28-13_05_46', 'rosbag2_2025_07_28-14_19_07', 'rosbag2_2025_10_17-13_19_29', 'rosbag2_2025_10_17-14_28_15', 'rosbag2_2025_10_17-15_11_00', 'rosbag2_2025_10_17-16_47_09'],
    "MainHallway": ['rosbag2_2025_10_09-10_23_38', 'rosbag2_2025_10_15-09_37_49', 'rosbag2_2025_10_15-11_03_05'],
    "Cafeteria": ['rosbag2_2025_07_07-10_49_31', 'rosbag2_2025_07_22-09_38_18', 'rosbag2_2025_07_22-10_59_25', 'rosbag2_2025_07_22-12_18_30', 'rosbag2_2025_07_22-13_30_39'],
    "EntranceCFacing": ['rosbag2_2025_10_07-15_03_48', 'rosbag2_2025_10_07-16_21_39', 'rosbag2_2025_10_09-17_37_23', 'rosbag2_2025_10_09-18_50_21', 'rosbag2_2025_10_15-12_27_14', 'rosbag2_2025_10_16-11_29_56'],
    "AstorPlace": ["2022_09_21_astor_place_landfill","2022_09_21_astor_place_recycle","2022_09_26_astor_place_landfill","2022_09_26_astor_place_recycle","2022_09_28_astor_place_landfill","2022_09_28_astor_place_recycle","2022_10_06_astor_place_landfill","2022_10_06_astor_place_recycle","2022_10_12_astor_place_landfill_0","2022_10_12_astor_place_landfill_1","2022_10_12_astor_place_recycle_0","2022_10_12_astor_place_recycle_1"],
    "AlbeeSquare": ["2023_07_06_albee_square_landfill_0","2023_07_06_albee_square_landfill_1","2023_07_06_albee_square_recycle_0","2023_07_06_albee_square_recycle_1","2023_07_07_albee_square_landfill_0","2023_07_07_albee_square_landfill_1","2023_07_07_albee_square_recycle_0","2023_07_07_albee_square_recycle_1","2023_07_11_albee_square_landfill_0","2023_07_11_albee_square_landfill_1","2023_07_11_albee_square_recycle_0","2023_07_11_albee_square_recycle_1","2023_07_12_albee_square_landfill_0","2023_07_12_albee_square_landfill_1","2023_07_12_albee_square_recycle_0","2023_07_12_albee_square_recycle_1","2023_07_14_albee_square_landfill","2023_07_14_albee_square_recycle"]
}

# lets do :
# - C104_vs_EntranceCBack
ROOM104_VS_ENTRANCECBACK_TRAIN = UNIQUE_PLACES_RECORDINGS["Room104"]
ROOM104_VS_ENTRANCECBACK_VAL = UNIQUE_PLACES_RECORDINGS["EntranceCBack"]

# - EntranceCBack_vs_C104
ENTRANCECBACK_VS_ROOM104_TRAIN = UNIQUE_PLACES_RECORDINGS["EntranceCBack"]
ENTRANCECBACK_VS_ROOM104_VAL = UNIQUE_PLACES_RECORDINGS["Room104"]

# - Cafeteria_vs_C104
CAFETERIA_VS_ROOM104_TRAIN = UNIQUE_PLACES_RECORDINGS["Cafeteria"]
CAFETERIA_VS_ROOM104_VAL = UNIQUE_PLACES_RECORDINGS["Room104"]

# - MainEntrance_vs_C104
MAINENTRANCE_VS_ROOM104_TRAIN = UNIQUE_PLACES_RECORDINGS["MainEntrance"]
MAINENTRANCE_VS_ROOM104_VAL = UNIQUE_PLACES_RECORDINGS["Room104"]

# - CoffeeB_vs_C104
COFFEEB_VS_ROOM104_TRAIN = UNIQUE_PLACES_RECORDINGS["CoffeeB"]
COFFEEB_VS_ROOM104_VAL = UNIQUE_PLACES_RECORDINGS["Room104"]

# - EntranceCBack_and_Cafetaria_vs_C104
ENTRANCECBACK_AND_CAFETERIA_VS_ROOM104_TRAIN = UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + UNIQUE_PLACES_RECORDINGS["Cafeteria"]
ENTRANCECBACK_AND_CAFETERIA_VS_ROOM104_VAL = UNIQUE_PLACES_RECORDINGS["Room104"]

# - EntranceCBack_and_Cafetaria_and_CoffeeB_vs_C104
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_ROOM104_TRAIN = UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + UNIQUE_PLACES_RECORDINGS["Cafeteria"] + UNIQUE_PLACES_RECORDINGS["CoffeeB"]
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_ROOM104_VAL = UNIQUE_PLACES_RECORDINGS["Room104"]

# - EntranceCBack_and_Cafetaria_and_CoffeeB_and_MainEntrance_vs_C104
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_ROOM104_TRAIN = UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + UNIQUE_PLACES_RECORDINGS["Cafeteria"] + UNIQUE_PLACES_RECORDINGS["CoffeeB"] + UNIQUE_PLACES_RECORDINGS["MainEntrance"]
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_ROOM104_VAL = UNIQUE_PLACES_RECORDINGS["Room104"]

# - all_vs_EntranceCBack
ALL_VS_ENTRANCECBACK_TRAIN = \
    UNIQUE_PLACES_RECORDINGS["Room005"] + \
    UNIQUE_PLACES_RECORDINGS["Bulle12X"] + \
    UNIQUE_PLACES_RECORDINGS["MainEntrance"] + \
    UNIQUE_PLACES_RECORDINGS["CoffeeB"] + \
    UNIQUE_PLACES_RECORDINGS["Room104"] + \
    UNIQUE_PLACES_RECORDINGS["MainHallway"] + \
    UNIQUE_PLACES_RECORDINGS["Cafeteria"] + \
    UNIQUE_PLACES_RECORDINGS["EntranceCFacing"]
ALL_VS_ENTRANCECBACK_VAL = UNIQUE_PLACES_RECORDINGS["EntranceCBack"]


# - all_vs_C104
ALL_VS_ROOM104_TRAIN = \
    UNIQUE_PLACES_RECORDINGS["Room005"] + \
    UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + \
    UNIQUE_PLACES_RECORDINGS["Bulle12X"] + \
    UNIQUE_PLACES_RECORDINGS["MainEntrance"] + \
    UNIQUE_PLACES_RECORDINGS["CoffeeB"] + \
    UNIQUE_PLACES_RECORDINGS["MainHallway"] + \
    UNIQUE_PLACES_RECORDINGS["Cafeteria"] + \
    UNIQUE_PLACES_RECORDINGS["EntranceCFacing"]
ALL_VS_ROOM104_VAL = UNIQUE_PLACES_RECORDINGS["Room104"]

# HUI_vs_AstorPlace
HUI_VS_ASTORPLACE_TRAIN = \
    UNIQUE_PLACES_RECORDINGS["Room005"] + \
    UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + \
    UNIQUE_PLACES_RECORDINGS["Bulle12X"] + \
    UNIQUE_PLACES_RECORDINGS["MainEntrance"] + \
    UNIQUE_PLACES_RECORDINGS["CoffeeB"] + \
    UNIQUE_PLACES_RECORDINGS["MainHallway"] + \
    UNIQUE_PLACES_RECORDINGS["Cafeteria"] + \
    UNIQUE_PLACES_RECORDINGS["EntranceCFacing"] + \
    UNIQUE_PLACES_RECORDINGS["Room104"]
HUI_VS_ASTORPLACE_VAL = UNIQUE_PLACES_RECORDINGS["AstorPlace"]

# HUI_vs_AlbeeSquare
HUI_VS_ALBEESQUARE_TRAIN = \
    UNIQUE_PLACES_RECORDINGS["Room005"] + \
    UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + \
    UNIQUE_PLACES_RECORDINGS["Bulle12X"] + \
    UNIQUE_PLACES_RECORDINGS["MainEntrance"] + \
    UNIQUE_PLACES_RECORDINGS["CoffeeB"] + \
    UNIQUE_PLACES_RECORDINGS["MainHallway"] + \
    UNIQUE_PLACES_RECORDINGS["Cafeteria"] + \
    UNIQUE_PLACES_RECORDINGS["EntranceCFacing"] + \
    UNIQUE_PLACES_RECORDINGS["Room104"]
HUI_VS_ALBEESQUARE_VAL = UNIQUE_PLACES_RECORDINGS["AlbeeSquare"]

# - EntranceCBack_vs_AlbeeSquare
ENTRANCECBACK_VS_ALBEESQUARE_TRAIN = UNIQUE_PLACES_RECORDINGS["EntranceCBack"]
ENTRANCECBACK_VS_ALBEESQUARE_VAL = UNIQUE_PLACES_RECORDINGS["AlbeeSquare"]

# - EntranceCBack_and_Cafetaria_vs_AlbeeSquare
ENTRANCECBACK_AND_CAFETERIA_VS_ALBEESQUARE_TRAIN = UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + UNIQUE_PLACES_RECORDINGS["Cafeteria"]
ENTRANCECBACK_AND_CAFETERIA_VS_ALBEESQUARE_VAL = UNIQUE_PLACES_RECORDINGS["AlbeeSquare"]

# - EntranceCBack_and_Cafetaria_and_CoffeeB_vs_AlbeeSquare
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_ALBEESQUARE_TRAIN = UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + UNIQUE_PLACES_RECORDINGS["Cafeteria"] + UNIQUE_PLACES_RECORDINGS["CoffeeB"]
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_ALBEESQUARE_VAL = UNIQUE_PLACES_RECORDINGS["AlbeeSquare"]

# - EntranceCBack_and_Cafetaria_and_CoffeeB_and_MainEntrance_vs_AlbeeSquare
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_ALBEESQUARE_TRAIN = UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + UNIQUE_PLACES_RECORDINGS["Cafeteria"] + UNIQUE_PLACES_RECORDINGS["CoffeeB"] + UNIQUE_PLACES_RECORDINGS["MainEntrance"]
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_ALBEESQUARE_VAL = UNIQUE_PLACES_RECORDINGS["AlbeeSquare"]

# - AstorPlace_vs_AlbeeSquare
ASTORPLACE_VS_ALBEESQUARE_TRAIN = UNIQUE_PLACES_RECORDINGS["AstorPlace"]
ASTORPLACE_VS_ALBEESQUARE_VAL = UNIQUE_PLACES_RECORDINGS["AlbeeSquare"]

# - HUI_and_AstorPlace_vs_AlbeeSquare
HUI_AND_ASTORPLACE_VS_ALBEESQUARE_TRAIN = \
    UNIQUE_PLACES_RECORDINGS["Room005"] + \
    UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + \
    UNIQUE_PLACES_RECORDINGS["Bulle12X"] + \
    UNIQUE_PLACES_RECORDINGS["MainEntrance"] + \
    UNIQUE_PLACES_RECORDINGS["CoffeeB"] + \
    UNIQUE_PLACES_RECORDINGS["MainHallway"] + \
    UNIQUE_PLACES_RECORDINGS["Cafeteria"] + \
    UNIQUE_PLACES_RECORDINGS["EntranceCFacing"] + \
    UNIQUE_PLACES_RECORDINGS["Room104"] + \
    UNIQUE_PLACES_RECORDINGS["AstorPlace"]
HUI_AND_ASTORPLACE_VS_ALBEESQUARE_VAL = UNIQUE_PLACES_RECORDINGS["AlbeeSquare"]


# - intra_C104_with_setups
INTRA_ROOM104_WITH_SETUPS_TRAIN = \
    ROOM104_SETUP_TO_RECORDINGS["Room104-1"] + \
    ROOM104_SETUP_TO_RECORDINGS["Room104-2"] + \
    ROOM104_SETUP_TO_RECORDINGS["Room104-3"] + \
    ROOM104_SETUP_TO_RECORDINGS["Room104-4"]
INTRA_ROOM104_WITH_SETUPS_VAL = ROOM104_SETUP_TO_RECORDINGS["Room104-5"]

# - intra_EntranceCBack_with_setups
INTRA_ENTRANCECBACK_WITH_SETUPS_TRAIN = \
    ENTRANCECBACK_SETUP_TO_RECORDINGS["EntranceCBack-1"] + \
    ENTRANCECBACK_SETUP_TO_RECORDINGS["EntranceCBack-2"] + \
    ENTRANCECBACK_SETUP_TO_RECORDINGS["EntranceCBack-3"] + \
    ENTRANCECBACK_SETUP_TO_RECORDINGS["EntranceCBack-4"]
INTRA_ENTRANCECBACK_WITH_SETUPS_VAL = \
    ENTRANCECBACK_SETUP_TO_RECORDINGS["EntranceCBack-5"] + \
    ENTRANCECBACK_SETUP_TO_RECORDINGS["EntranceCBack-6"] + \
    ENTRANCECBACK_SETUP_TO_RECORDINGS["EntranceCBack-7"] + \
    ENTRANCECBACK_SETUP_TO_RECORDINGS["EntranceCBack-8"]
    


########### DEFINE SOME ENSEMBLES ################
HUI_TRAININGSET = \
    UNIQUE_PLACES_RECORDINGS["Room005"] + \
    UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + \
    UNIQUE_PLACES_RECORDINGS["Bulle12X"] + \
    UNIQUE_PLACES_RECORDINGS["MainEntrance"] + \
    UNIQUE_PLACES_RECORDINGS["CoffeeB"] + \
    UNIQUE_PLACES_RECORDINGS["Cafeteria"] + \
    UNIQUE_PLACES_RECORDINGS["MainHallway"]

HUI_TESTINGSET= \
    UNIQUE_PLACES_RECORDINGS["Room104"] + \
    UNIQUE_PLACES_RECORDINGS["EntranceCFacing"]
    
SSUP_TRAININGSET = UNIQUE_PLACES_RECORDINGS["AstorPlace"]
SSUP_TESTINGSET = UNIQUE_PLACES_RECORDINGS["AlbeeSquare"]


# EntranceCBack_vs_SSUPTesting
# EntranceCBack_and_Cafeteria_vs_SSUPTesting
# EntranceCBack_and_Cafeteria_and_CoffeeB_vs_SSUPTesting
# EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_vs_SSUPTesting
# EntanceCBack_vs_HUITesting
# EntranceCBack_and_Cafeteria_vs_HUITesting
# EntranceCBack_and_Cafeteria_and_CoffeeB_vs_HUITesting
# EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_vs_HUITesting
# Cafeteria_vs_HUITesting
# CoffeeB_vs_HUITesting
# MainEntrance_vs_HUITesting

# Now other kind of cross
# - EntranceCBack_vs_SSUPTesting
ENTRANCECBACK_VS_SSUPTESTING_TRAIN = UNIQUE_PLACES_RECORDINGS["EntranceCBack"]
ENTRANCECBACK_VS_SSUPTESTING_VAL = SSUP_TESTINGSET

# - EntranceCBack_and_Cafeteria_vs_SSUPTesting
ENTRANCECBACK_AND_CAFETERIA_VS_SSUPTESTING_TRAIN = \
    UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + \
    UNIQUE_PLACES_RECORDINGS["Cafeteria"]
ENTRANCECBACK_AND_CAFETERIA_VS_SSUPTESTING_VAL = SSUP_TESTINGSET

# - EntranceCBack_and_Cafeteria_and_CoffeeB_vs_SSUPTesting
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_SSUPTESTING_TRAIN = \
    UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + \
    UNIQUE_PLACES_RECORDINGS["Cafeteria"] + \
    UNIQUE_PLACES_RECORDINGS["CoffeeB"]
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_SSUPTESTING_VAL = SSUP_TESTINGSET

# - EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_vs_SSUPTesting
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_SSUPTESTING_TRAIN = \
    UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + \
    UNIQUE_PLACES_RECORDINGS["Cafeteria"] + \
    UNIQUE_PLACES_RECORDINGS["CoffeeB"] + \
    UNIQUE_PLACES_RECORDINGS["MainEntrance"]
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_SSUPTESTING_VAL = SSUP_TESTINGSET

# - EntanceCBack_vs_HUITesting
ENTRANCECBACK_VS_HUITESTING_TRAIN = UNIQUE_PLACES_RECORDINGS["EntranceCBack"]
ENTRANCECBACK_VS_HUITESTING_VAL = HUI_TESTINGSET

# - EntranceCBack_and_Cafeteria_vs_HUITesting
ENTRANCECBACK_AND_CAFETERIA_VS_HUITESTING_TRAIN = \
    UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + \
    UNIQUE_PLACES_RECORDINGS["Cafeteria"]
ENTRANCECBACK_AND_CAFETERIA_VS_HUITESTING_VAL = HUI_TESTINGSET

# - EntranceCBack_and_Cafeteria_and_CoffeeB_vs_HUITesting
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_HUITESTING_TRAIN = \
    UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + \
    UNIQUE_PLACES_RECORDINGS["Cafeteria"] + \
    UNIQUE_PLACES_RECORDINGS["CoffeeB"]
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_HUITESTING_VAL = HUI_TESTINGSET

# - EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_vs_HUITesting
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_HUITESTING_TRAIN = \
    UNIQUE_PLACES_RECORDINGS["EntranceCBack"] + \
    UNIQUE_PLACES_RECORDINGS["Cafeteria"] + \
    UNIQUE_PLACES_RECORDINGS["CoffeeB"] + \
    UNIQUE_PLACES_RECORDINGS["MainEntrance"]
ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_HUITESTING_VAL = HUI_TESTINGSET

# - Cafeteria_vs_HUITesting
CAFETERIA_VS_HUITESTING_TRAIN = UNIQUE_PLACES_RECORDINGS["Cafeteria"]
CAFETERIA_VS_HUITESTING_VAL = HUI_TESTINGSET

# - CoffeeB_vs_HUITesting
COFFEEB_VS_HUITESTING_TRAIN = UNIQUE_PLACES_RECORDINGS["CoffeeB"]
COFFEEB_VS_HUITESTING_VAL = HUI_TESTINGSET

# - MainEntrance_vs_HUITesting
MAINENTRANCE_VS_HUITESTING_TRAIN = UNIQUE_PLACES_RECORDINGS["MainEntrance"]
MAINENTRANCE_VS_HUITESTING_VAL = HUI_TESTINGSET


# HUI360train = HUI sans 7 et 9
# HUI360Test = HUI 7 et 9
# SSUPTrain = 10
# SSUPTest = 11
# FullTrain = SSUPTrain + HUITrain
# \textcolor{blue}{Add recap table with best result : intra-HUI360, BestOnAlbeeSquare (en testant avec HUI-7 et 9), Beston7et9, BestOnAlbeeSuquare (avec 10). Avec AUC et F1 et petite justif de pourquoi AUC dans les  ablations}

START_EXPES_IDS = 100 # no expe under 100

def main(args):

    EXPERIMENTS_CONF_DIR = os.path.join(here, "..", "experiments", "generated_configs")
    if not os.path.isdir(EXPERIMENTS_CONF_DIR):
        prWarning(f"Creating {EXPERIMENTS_CONF_DIR} as it did not existed")
        os.makedirs(EXPERIMENTS_CONF_DIR)

    experiments_dirs = [d for d in os.walk(EXPERIMENTS_CONF_DIR)][0][1]
    next_expe_id = START_EXPES_IDS
    if len(experiments_dirs) > 0:
        listed_expe = [e for e in experiments_dirs if e.startswith("expe_")]
        listed_expe.sort()
        # next_expe_id = len(listed_expe)
        while f"expe_{next_expe_id:03d}" in listed_expe:
            next_expe_id += 1

    prWarning(f"Using next experiment id : {next_expe_id}")
            
    expe_path = os.path.join(EXPERIMENTS_CONF_DIR, "expe_{:03d}".format(next_expe_id))
    if not os.path.exists(expe_path):
        os.makedirs(expe_path)
    else:
        prError(f"Directory {expe_path} already exists")
        exit(0)        

    SEARCH_DIC = {
        # Directory related hyperparameters
        # "experiment_name": f"expe_{next_expe_id:03d}", # not searched but assigned
        # "hostname": f"GeneratedConfigOn{hostname}", # not searched but fixed
        "force_model_type": ["mb_from_pretrained_finetune"], #, "mb_from_pretrained"], #"stg_nf", "stgcn", "mlp", "lstm", "skateformer", "mb_from_pretrained_finetune", "mb_from_pretrained", "mb_lite_from_pretrained_finetune", "mb_lite_from_pretrained"], #["stg_nf", "stgcn", "mlp", "lstm", "skateformer"], #"mb_from_pretrained_finetune", "mb_from_pretrained", "mb_lite_from_pretrained_finetune", "mb_lite_from_pretrained"], #, "stgcn", "mlp", "skateformer"], #, "stgcn", "lstm", "skateformer"],
        
        "weight_decay": [0.0, 0.00005], #[0.1, 0.00005, 0.0], # 0.00005

        # General related hyperparameters
        "learning_rate": [0.01, 0.001, 0.0001], #[0.001],
        "lr_scheduler_type": ["none"], #"none", "CosineAnnealingWarmRestarts"], # "CosineAnnealingWarmRestarts", "ExponentialDecay", CosineAnnealingWithWarmup
        
        # NOTE : lr_decay is used as a list with parameters for CosineAnnealingWarmRestarts (3 elements : initial_cycle_T, cycle_T_mult, eta_min) and CosineAnnealingWithWarmup (6 elements : SAME + cycle_T_decay, initial_lr_before_warmup, warmup_steps). If initial_cycle_T == -1, then will use all epochs
        "lr_decay": [0.99, [5, 2, 0.00001], [-1, 1, 0.00001, 1, 0.0000001, 25], [25, 1.5, 0.00001, 0.75, 0.0000001, 25]], #, [10, 1.5, 0.00001, 0.75, 0.0000001, 10]], #0.99
        
        "loss_type": ["BCEWithLogitsLoss"],
        "use_weighted_loss": [True],
        "optimizer_type": ["AdamW"], # AdamW
        "grad_clip": [1.0, -1.0], # [-1.0, 1.0, 100.0], # -1.0 mean no grad clip

        "batch_size": [64],
        "epochs": [600], #, 100, 600],
        
        # MB related hyperparameters
        "mb_input_norm": ["vid"], # "scale" (ie crop_scale_torch with [1,1]), "scale_sample" (ie crop_scale_torch_by_sample) or "vid" (ie use video size for normalization). In the first case mb requires normalize_in_image to be True, in the second case it requires normalize_in_image to be False.
        "mb_head_dropout": [0.5],
        "mb_head_hidden_dim": [2048],
        "mb_head_version": ["class_time_avg"],
        "mb_desired_return": ["representation"],
        
        # MLP related hyperparameters
        "hidden_dims": [[32, 256]],
        "dropout": [0.0],
        
        # LSTM related hyperparameters
        "lstm_hidden_dim": [128],
        "lstm_num_layers": [3],
        "lstm_dropout": [0.0],
        
        # RF Training related hyperparameters
        "n_estimators": [500],
        "max_depth": [5],
        "class_weight": ['balanced'],
        
        # STG-NF related hyperparameters
        "stg_nf_hidden_channels": [16],
        "stg_nf_K": [16],
        "stg_nf_L": [1],
        "stg_nf_R": [10.0],
        "stg_nf_actnorm_scale": [1.0],
        "stg_nf_edge_importance": [False],
        "stg_nf_max_hops": [16],
        
        # STGCN related hyperparameters
        "stgcn_in_channels": [3],
        "stgcn_edge_importance_weighting": [True],
        "stgcn_layout": ["openpose"],
        
        # SkateFormer related hyperparameters
        "skateformer_in_channels": [3],
        
        # Dataset related hyperparameters
        # "train_tracks_filename": ["tracks_train_2025_10_31_20_15_49_fold_0.txt", "tracks_train_2025_10_31_20_15_49_fold_1.txt", "tracks_train_2025_10_31_20_15_49_fold_2.txt", "tracks_train_2025_10_31_20_15_49_fold_3.txt", "tracks_train_2025_10_31_20_15_49_fold_4.txt"],
        # "val_tracks_filename": ["tracks_val_2025_10_31_20_15_49_fold_0.txt", "tracks_val_2025_10_31_20_15_49_fold_1.txt", "tracks_val_2025_10_31_20_15_49_fold_2.txt", "tracks_val_2025_10_31_20_15_49_fold_3.txt", "tracks_val_2025_10_31_20_15_49_fold_4.txt"],
        # "test_tracks_filename": ["tracks_test_2025_10_31_20_15_49_fold_0.txt", "tracks_test_2025_10_31_20_15_49_fold_1.txt", "tracks_test_2025_10_31_20_15_49_fold_2.txt", "tracks_test_2025_10_31_20_15_49_fold_3.txt", "tracks_test_2025_10_31_20_15_49_fold_4.txt"],
        # "fold": ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"],
        # "train_tracks_filename": ["tracks_train_2025_10_31_20_15_49_fold"],
        # "val_tracks_filename": ["tracks_val_2025_10_31_20_15_49_fold"],
        # "test_tracks_filename": ["tracks_test_2025_10_31_20_15_49_fold"],
        
        
        "train_tracks_filename": ["all"], # ["train_hui_train_vs_hui_test_20260125_194459.txt"],
        "val_tracks_filename": ["all"], # ["val_hui_train_vs_hui_test_20260125_194459.txt"],
        "test_tracks_filename": ["all"], # ["val_hui_train_vs_hui_test_20260125_194459.txt"],
        
        # When changing time or tADV (keep under tADV=30 and time=30)
        # "train_tracks_filename": ["2025_11_11_18_15_29_tracks_with_valid_input_len_30_int_30_pos_30_force_pos_True_ignore_neg_True_force_align_True.txt"],
        # "val_tracks_filename": ["2025_11_11_18_15_29_tracks_with_valid_input_len_30_int_30_pos_30_force_pos_True_ignore_neg_True_force_align_True.txt"],
        # "test_tracks_filename": ["2025_11_11_18_15_29_tracks_with_valid_input_len_30_int_30_pos_30_force_pos_True_ignore_neg_True_force_align_True.txt"],
        
        # "train_tracks_filename": ["tracks_train_2025_10_28_13_22_44.txt"],
        # "val_tracks_filename": ["tracks_val_2025_10_28_13_22_44.txt"],
        # "test_tracks_filename": ["tracks_test_2025_10_28_13_22_44.txt"],
        "positive_cutoff_train": [15,30], #[0, 5, 15, 20, 25, 30], # equiv to T_{POS}, i.e. the number of frames before the interaction at which point we start putting the label 1
        "interaction_cutoff_train": [0,15,30], #[0, 5, 15, 20, 25, 30], # equiv to T_{CUT}, i.e. the number of frames before the interaction at which point we discard the segment (too close to the interaction, thus too "easy") | NB: it is also used for the onset aligmenet with biggest mask size
        "positive_cutoff_val": [15], #[0, 5, 15, 20, 25, 30], # equiv to T_{POS}
        "interaction_cutoff_val": [15], #[0, 5, 15, 20, 25, 30], # equiv to T_{CUT}
        
        # this is a way of using a larger fixed input length and then during training only using a bit of it. Use with caution... 
        # This limits the "available" tracks by only taking those with the maximum input_iength_in_frames + cutoffs, but is not compatible with track normalization techniques
        # [0,-1] means use all frames
        
        "fixed_input_length": [True],
        "input_length_in_frames": [30], #30 #[24],
        "subsample_frames": [1],
        "fix_index_per_track_train": [True], # [True, False],
        "fix_index_per_track_list_train": [FIX_LIST], # [FIX_LIST, None],
        "fix_index_per_track_val": [True],
        "fix_index_per_track_list_val": [FIX_LIST],
        "min_length_in_frames": [None],
        "max_length_in_frames": [None],
        "min_keypoints_filter": [9],
        "additional_filtering_dict": [{"mask_size": {"min": 1000, "max": 1e7}}],
        "normalize_in_image": [True, False],
        "normalize_keypoints_in_box": [True, False],
        "normalize_keypoints_in_track": ["none"], #, "norm_xy", "norm_x"],
        "do_recenter_interaction_zone": [True], #should be [True],
        "standardize_data": ["none", "all"], # ["all", "none"], # "none", "mask_only"
        "use_polar_coordinates": [False],
        "force_positive_samples": [True],
        "force_aligment_with_biggest_mask_size_train": [True], #[True],
        "force_aligment_with_biggest_mask_size_val": [True], #[True],
        "center_on_onset_train": [False],
        "center_on_onset_val": [False],
        
        "random_flip_horizontal_train": [True], # False
        "random_flip_horizontal_val": [False],
        "random_jitter_position_train": [[0.0, 0.0]],
        "random_jitter_position_val": [[0.0, 0.0]],
        
        "do_recentering_train": [False],
        "do_recentering_val": [False],
        "do_fix_keypoints_outside_box_train": [True],
        "do_fix_keypoints_outside_box_val": [True],
        
        "inputs_per_track_stride_train": [-1], #[-1], # [6],
        "inputs_per_track_stride_val": [-1],
        
        "include_recordings_train": [
            HUI_TRAININGSET,
            # SSUP_TRAININGSET,
            ], 
        
        "include_recordings_val": [
            HUI_TESTINGSET,
            # SSUP_TESTINGSET,
            ],
        
        "include_columns" : [FEATURES_SET_D3],
        
        "hf_dataset_revision": ["main"], #main #3c8a342548534b6b92d32b0099e266962facdf45
        
        "comment": ["BaselinesForFuturePretraining"],
        
    }


    indexes = []
    for k, vl in SEARCH_DIC.items():
        # print(k, len(vl))
        assert(type(vl) == list)
        index_hp = [i for i in range(len(vl))]
        indexes.append(index_hp)
        
    combinations = list(itertools.product(*indexes))
    prInfo("Got {} combinations of parameters".format(len(combinations)))

    all_dics = {}
    for combcount, comb in enumerate(combinations):
        comb_dic = {}
        for idhp, (k, vl) in enumerate(SEARCH_DIC.items()):
            comb_dic[k] = vl[comb[idhp]]
        
        valid_comb = True
    
    
        ############################## TEMPORARY RESTRICTIONS ##############################
        
        if (comb_dic["force_model_type"] == "lstm" or comb_dic["force_model_type"] == "mlp") and comb_dic["normalize_keypoints_in_box"] == False:
            # works better with normalized keypoints in box
            valid_comb = False
            
        if (comb_dic["force_model_type"] != "lstm" and comb_dic["force_model_type"] != "mlp") and comb_dic["normalize_keypoints_in_box"] == True:
            # works better without normalized keypoints in box (other models do not have box coordinates !)
            valid_comb = False
        
        #####################################################################################
    
        
        ####### FORCE ALL CUTOFFS TO BE THE SAME ##########
        # if comb_dic["positive_cutoff_val"] != comb_dic["interaction_cutoff_val"]:
        #     # To evaluate with a precise cutoff
        #     valid_comb = False
        # if comb_dic["positive_cutoff_train"] != comb_dic["interaction_cutoff_train"]:
        #     # To evaluate with a precise cutoff
        #     valid_comb = False
        # if comb_dic["positive_cutoff_val"] != comb_dic["positive_cutoff_train"]:
        #     # To evaluate with a precise cutoff
        #     valid_comb = False
        # if comb_dic["interaction_cutoff_train"] != comb_dic["interaction_cutoff_val"]:
        #     # To evaluate with a precise cutoff
        #     valid_comb = False
        ######################################
        
        # For training : does not make sens to have a bigger interaction_cutoff than positive_cutoff (for val anyway they must be the same)
        if comb_dic["positive_cutoff_train"] < comb_dic["interaction_cutoff_train"]:
            valid_comb = False

        # No point in changing the cutoffs if anyway the sample is centered on the onset
        if comb_dic["center_on_onset_train"] and (comb_dic["positive_cutoff_train"] != SEARCH_DIC["positive_cutoff_train"][0]):
            valid_comb = False
        if comb_dic["center_on_onset_train"] and (comb_dic["interaction_cutoff_train"] != SEARCH_DIC["interaction_cutoff_train"][0]):
            valid_comb = False
        if comb_dic["center_on_onset_val"] and (comb_dic["positive_cutoff_val"] != SEARCH_DIC["positive_cutoff_val"][0]):
            valid_comb = False
        if comb_dic["center_on_onset_val"] and (comb_dic["interaction_cutoff_val"] != SEARCH_DIC["interaction_cutoff_val"][0]):
            valid_comb = False
            

        # Make the alignement strategy consistent
        if comb_dic["force_aligment_with_biggest_mask_size_train"] != comb_dic["force_aligment_with_biggest_mask_size_val"]:
            # either both are False or both are True
            valid_comb = False

        if comb_dic["center_on_onset_train"] and comb_dic["force_aligment_with_biggest_mask_size_train"]:
            # cannot do both at the same time
            valid_comb = False
        if comb_dic["center_on_onset_val"] and comb_dic["force_aligment_with_biggest_mask_size_val"]:
            # cannot do both at the same time
            valid_comb = False
            
        ### For graph models only some features sets are allowed (D3 converted to NW-UCLA and D9 converted to NTU)
        if (comb_dic["force_model_type"] == "skateformer") and (comb_dic["include_columns"] not in [FEATURES_SET_D3, FEATURES_SET_D9]):
            valid_comb = False

        ### For graph models only some features sets are allowed
        if (comb_dic["force_model_type"] == "stgcn") and (comb_dic["include_columns"] not in [FEATURES_SET_D3]):
            valid_comb = False
            
        ########### Use dedicated learning rates for each model type ##########
        if comb_dic["learning_rate"] != 0.001 and comb_dic["force_model_type"] == "lstm":
            valid_comb = False
        if comb_dic["learning_rate"] != 0.01 and comb_dic["force_model_type"] == "stg_nf":
            valid_comb = False
        if comb_dic["learning_rate"] != 0.0001 and "mb_" in comb_dic["force_model_type"]:
            valid_comb = False
        if comb_dic["learning_rate"] != 0.001 and comb_dic["force_model_type"] == "skateformer":
            valid_comb = False
        if comb_dic["learning_rate"] != 0.001 and comb_dic["force_model_type"] == "mlp":
            valid_comb = False
        if comb_dic["learning_rate"] != 0.001 and comb_dic["force_model_type"] == "stgcn":
            valid_comb = False

        ########### Use dedicated weight decay for each model type ##########
        if comb_dic["weight_decay"] != 0.00005 and comb_dic["force_model_type"] == "lstm":
            valid_comb = False
        if comb_dic["weight_decay"] != 0.0 and comb_dic["force_model_type"] == "stg_nf":
            valid_comb = False
        if comb_dic["weight_decay"] != 0.00005 and "mb_" in comb_dic["force_model_type"]:
            valid_comb = False
        if comb_dic["weight_decay"] != 0.1 and comb_dic["force_model_type"] == "skateformer":
            valid_comb = False
        if comb_dic["weight_decay"] != 0.0 and comb_dic["force_model_type"] == "mlp":
            valid_comb = False
        if comb_dic["weight_decay"] != 0.1 and comb_dic["force_model_type"] == "stgcn":
            valid_comb = False

        ###### Use dedicated number of epochs for each model type ##########
        if comb_dic["epochs"] != 100 and comb_dic["force_model_type"] == "lstm":
            valid_comb = False
        if comb_dic["epochs"] != 600 and comb_dic["force_model_type"] == "stg_nf":
            valid_comb = False
        if comb_dic["epochs"] != 600 and "mb_" in comb_dic["force_model_type"]:
            valid_comb = False
        if comb_dic["epochs"] != 100 and comb_dic["force_model_type"] == "mlp":
            valid_comb = False
        if comb_dic["epochs"] != 100 and comb_dic["force_model_type"] == "stgcn":
            valid_comb = False
        if comb_dic["epochs"] != 600 and comb_dic["force_model_type"] == "skateformer":
            valid_comb = False
            
        # gradient clipping only for skateformer
        if (comb_dic["force_model_type"] != "skateformer" and comb_dic["force_model_type"] != "stg_nf") and (comb_dic["grad_clip"] != -1.0):
            valid_comb = False
        if comb_dic["grad_clip"] != 1.0 and comb_dic["force_model_type"] == "skateformer":
            valid_comb = False
        if comb_dic["grad_clip"] != 100.0 and comb_dic["force_model_type"] == "stg_nf":
            valid_comb = False
            
        # Normalization constraints
        if comb_dic["normalize_keypoints_in_box"] and comb_dic["normalize_in_image"] == False:
            # normalize in box requires normalize in image
            valid_comb = False
        if comb_dic["normalize_keypoints_in_box"] and comb_dic["normalize_keypoints_in_track"] != "none":
            # if already normalized in box, does not make sense to normalize in track
            valid_comb = False
            
        # Specific normalization constraints for mb models
        if ("mb_" in comb_dic["force_model_type"]) and (comb_dic["mb_input_norm"] == "scale" or comb_dic["mb_input_norm"] == "scale_sample") and comb_dic["normalize_in_image"] == False:
            # scale (outdated) and scale_sample both require normalize in image to be False
            valid_comb = False
        if ("mb_" in comb_dic["force_model_type"]) and (comb_dic["mb_input_norm"] == "vid") and comb_dic["normalize_in_image"] == True:
            # vid requires normalize in image to be True
            valid_comb = False
        if comb_dic["normalize_keypoints_in_track"] != "none" and "mb_" in comb_dic["force_model_type"]:
            # no track normalization for mb
            valid_comb = False
        if comb_dic["normalize_keypoints_in_box"] and "mb_" in comb_dic["force_model_type"]:
            # no box normalization for mb
            valid_comb = False
                
        # For all other models we alway normalize in image (at least)   
        if ("mb_" not in comb_dic["force_model_type"]) and comb_dic["normalize_in_image"] == False:
            valid_comb = False

        # Some standardization required for those models. Note than "mask_only" standardization is allowed for all models.
        if comb_dic["standardize_data"] == "none" and (comb_dic["force_model_type"] == "lstm" or comb_dic["force_model_type"] == "mlp"):
            # MLP and LSTM require at least mask_size to be standardized
            valid_comb = False
        if comb_dic["standardize_data"] == "all" and (comb_dic["force_model_type"] == "stg_nf" or "mb_" in comb_dic["force_model_type"] or comb_dic["force_model_type"] == "skateformer" or comb_dic["force_model_type"] == "stgcn"):
            # In STG-NF, SkateFormer, STGCN and MB skeleton data should not be standardized
            valid_comb = False
        
        # Condition to use polar coordinates
        if comb_dic["use_polar_coordinates"] == True and comb_dic["force_model_type"] != "lstm":
            valid_comb = False
        if comb_dic["use_polar_coordinates"] == True and comb_dic["normalize_keypoints_in_track"] != "none":
            # No track normalization allowed when using polar coordinates
            valid_comb = False
        if comb_dic["use_polar_coordinates"] == True and comb_dic["normalize_keypoints_in_box"] == True:
            # No box normalization allowed when using polar coordinates
            valid_comb = False
        if comb_dic["use_polar_coordinates"] == True and comb_dic["normalize_in_image"] == False:
            # Need image normalization when using polar coordinates (0-1 converted to angles)
            valid_comb = False
        if comb_dic["use_polar_coordinates"] == True and comb_dic["standardize_data"] == "all":
            # No skeleton standardization allowed when using polar coordinates
            valid_comb = False
        
        # Learning rate scheduler constraints
        if comb_dic["lr_scheduler_type"] == "CosineAnnealingWarmRestarts" and (type(comb_dic["lr_decay"]) != list or (type(comb_dic["lr_decay"]) == list and len(comb_dic["lr_decay"]) != 3)):
            valid_comb = False
        if comb_dic["lr_scheduler_type"] == "CosineAnnealingWithWarmup" and (type(comb_dic["lr_decay"]) != list or (type(comb_dic["lr_decay"]) == list and len(comb_dic["lr_decay"]) != 6)):
            valid_comb = False
        if comb_dic["lr_scheduler_type"] == "ExponentialDecay" and type(comb_dic["lr_decay"]) != float:
            valid_comb = False
        if comb_dic["lr_scheduler_type"] == "none" and comb_dic["lr_decay"] != SEARCH_DIC["lr_decay"][0]:
            # no need to change decay if no scheduler
            valid_comb = False
            
        # if not using a single input per track there is no point using fix_index_per_track (this is not used in the dataloader in this case)
        if comb_dic["inputs_per_track_stride_train"] != -1 and (comb_dic["fix_index_per_track_train"] or comb_dic["fix_index_per_track_list_train"] is not None):
            valid_comb = False
        if comb_dic["inputs_per_track_stride_val"] != -1 and (comb_dic["fix_index_per_track_val"] or comb_dic["fix_index_per_track_list_val"] is not None):
            valid_comb = False
        
        # assert folding is coherent
        if comb_dic["train_tracks_filename"].endswith("fold"):
            assert(comb_dic["val_tracks_filename"].endswith("fold") and comb_dic["test_tracks_filename"].endswith("fold"))
        else:
            if comb_dic["train_tracks_filename"] == "all" and comb_dic["val_tracks_filename"] == "all" and comb_dic["test_tracks_filename"] == "all":
                pass # ok this is the cross
            else:
                # pass
                assert(comb_dic["val_tracks_filename"].endswith(".txt") and comb_dic["test_tracks_filename"].endswith(".txt") and comb_dic["train_tracks_filename"].endswith(".txt"))

        # to train and test with the same do_recentering and do_fix_keypoints_outside_box
        if comb_dic["do_recentering_train"] != comb_dic["do_recentering_val"]:
            valid_comb = False
        if comb_dic["do_fix_keypoints_outside_box_train"] != comb_dic["do_fix_keypoints_outside_box_val"]:
            valid_comb = False
        
        # if model type is forced do search for hyperparameters for the other model type
        # ignore MLP hyperparameters for other model types
        if comb_dic["force_model_type"] != "mlp" and (comb_dic["hidden_dims"] != SEARCH_DIC["hidden_dims"][0] 
                                                      or comb_dic["dropout"] != SEARCH_DIC["dropout"][0]):
            valid_comb = False
        
        # ignore RF hyperparameters for other model types
        if comb_dic["force_model_type"] != "rf" and (comb_dic["n_estimators"] != SEARCH_DIC["n_estimators"][0]
                                                      or comb_dic["max_depth"] != SEARCH_DIC["max_depth"][0]
                                                      or comb_dic["class_weight"] != SEARCH_DIC["class_weight"][0]):
            valid_comb = False

        # ignore LSTM hyperparameters for other model types
        if comb_dic["force_model_type"] != "lstm" and (comb_dic["lstm_hidden_dim"] != SEARCH_DIC["lstm_hidden_dim"][0] 
                                                      or comb_dic["lstm_num_layers"] != SEARCH_DIC["lstm_num_layers"][0] 
                                                      or comb_dic["lstm_dropout"] != SEARCH_DIC["lstm_dropout"][0]):
            valid_comb = False
        
        # ignore STG-NF hyperparameters for other model types
        if comb_dic["force_model_type"] != "stg_nf" and (comb_dic["stg_nf_K"] != SEARCH_DIC["stg_nf_K"][0]
                                                         or comb_dic["stg_nf_L"] != SEARCH_DIC["stg_nf_L"][0]
                                                         or comb_dic["stg_nf_R"] != SEARCH_DIC["stg_nf_R"][0]
                                                         or comb_dic["stg_nf_actnorm_scale"] != SEARCH_DIC["stg_nf_actnorm_scale"][0]
                                                         or comb_dic["stg_nf_edge_importance"] != SEARCH_DIC["stg_nf_edge_importance"][0]
                                                         or comb_dic["stg_nf_max_hops"] != SEARCH_DIC["stg_nf_max_hops"][0]
                                                         or comb_dic["stg_nf_hidden_channels"] != SEARCH_DIC["stg_nf_hidden_channels"][0]):
            # ignore stgnf parameters for other model types
            valid_comb = False

        # ignore MB hyperparameters for other model types
        if "mb_" not in comb_dic["force_model_type"] and (comb_dic["mb_input_norm"] != SEARCH_DIC["mb_input_norm"][0]
                                                         or comb_dic["mb_head_dropout"] != SEARCH_DIC["mb_head_dropout"][0]
                                                         or comb_dic["mb_head_hidden_dim"] != SEARCH_DIC["mb_head_hidden_dim"][0]
                                                         or comb_dic["mb_head_version"] != SEARCH_DIC["mb_head_version"][0]):
            # ignore mb parameters for other model types
            valid_comb = False
            
        # ignore STGCN hyperparameters for other model types
        if comb_dic["force_model_type"] != "stgcn" and (comb_dic["stgcn_in_channels"] != SEARCH_DIC["stgcn_in_channels"][0]
                                                         or comb_dic["stgcn_edge_importance_weighting"] != SEARCH_DIC["stgcn_edge_importance_weighting"][0]
                                                         or comb_dic["stgcn_layout"] != SEARCH_DIC["stgcn_layout"][0]):
            # ignore stgcn parameters for other model types
            valid_comb = False


        # ignore SkateFormer hyperparameters for other model types
        if comb_dic["force_model_type"] != "skateformer" and (comb_dic["skateformer_in_channels"] != SEARCH_DIC["skateformer_in_channels"][0]):
            # ignore skateformer parameters for other model types
            valid_comb = False

        
        # Reproducitbility compatibility
        if comb_dic["fix_index_per_track_train"] == True and comb_dic["fix_index_per_track_list_train"] == None:
            # if fix do it fully
            valid_comb = False
        if comb_dic["fix_index_per_track_train"] == False and comb_dic["fix_index_per_track_list_train"] != None:
            # if not fix, then list must be None
            valid_comb = False

        if comb_dic["fix_index_per_track_val"] == True and comb_dic["fix_index_per_track_list_val"] == None:
            # if fix do it fully
            valid_comb = False
        if comb_dic["fix_index_per_track_val"] == False and comb_dic["fix_index_per_track_list_val"] != None:
            # if not fix, then list must be None
            valid_comb = False
            

        # Name the aligment strategy
        if comb_dic["force_aligment_with_biggest_mask_size_train"] == True:
            comb_dic["force_align_negatives_train"] = "force_aligment"
        else:
            comb_dic["force_align_negatives_train"] = "none"

        if comb_dic["force_aligment_with_biggest_mask_size_val"] == True:
            comb_dic["force_align_negatives_val"] = "force_aligment"
        else:
            comb_dic["force_align_negatives_val"] = "none"
    
        
        ########### SET A CROSS EVALUATION NAME ###########


        # add a name, note that we need to convert the string to the actual list for the training and validation (using strings instead of directly lists avoids confusions)
        if comb_dic["include_recordings_train"] == "all" and comb_dic["include_recordings_val"] == "all":
            comb_dic["cross_eval_type"] = "full_hui"
        
        elif comb_dic["include_recordings_train"] == HUI_TRAININGSET and comb_dic["include_recordings_val"] == HUI_TESTINGSET:
            comb_dic["cross_eval_type"] = "hui_train_vs_hui_test"
            comb_dic["include_recordings_train"] = HUI_TRAININGSET
            comb_dic["include_recordings_val"] = HUI_TESTINGSET
        elif comb_dic["include_recordings_train"] == SSUP_TRAININGSET and comb_dic["include_recordings_val"] == SSUP_TESTINGSET:
            comb_dic["cross_eval_type"] = "ssup_train_vs_ssup_test"
            comb_dic["include_recordings_train"] = SSUP_TRAININGSET
            comb_dic["include_recordings_val"] = SSUP_TESTINGSET
        elif comb_dic["include_recordings_train"] == HUI_TRAININGSET and comb_dic["include_recordings_val"] == SSUP_TESTINGSET:
            comb_dic["cross_eval_type"] = "hui_train_vs_ssup_test"
            comb_dic["include_recordings_train"] = HUI_TRAININGSET
            comb_dic["include_recordings_val"] = SSUP_TESTINGSET
        elif comb_dic["include_recordings_train"] == SSUP_TRAININGSET and comb_dic["include_recordings_val"] == HUI_TESTINGSET:
            comb_dic["cross_eval_type"] = "ssup_train_vs_hui_test"
            comb_dic["include_recordings_train"] = SSUP_TRAININGSET
            comb_dic["include_recordings_val"] = HUI_TESTINGSET
        elif comb_dic["include_recordings_train"] == HUI_TESTINGSET and comb_dic["include_recordings_val"] == HUI_TRAININGSET:
            comb_dic["cross_eval_type"] = "reverse_testtrain_hui_train_vs_hui_test"
            comb_dic["include_recordings_train"] = HUI_TESTINGSET
            comb_dic["include_recordings_val"] = HUI_TRAININGSET
        elif comb_dic["include_recordings_train"] == SSUP_TESTINGSET and comb_dic["include_recordings_val"] == SSUP_TRAININGSET:
            comb_dic["cross_eval_type"] = "reverse_testtrain_ssup_train_vs_ssup_test"
            comb_dic["include_recordings_train"] = SSUP_TESTINGSET
            comb_dic["include_recordings_val"] = SSUP_TRAININGSET
        elif comb_dic["include_recordings_train"] == HUI_TESTINGSET and comb_dic["include_recordings_val"] == SSUP_TRAININGSET:
            comb_dic["cross_eval_type"] = "reverse_testtrain_hui_train_vs_ssup_test"
            comb_dic["include_recordings_train"] = HUI_TESTINGSET
            comb_dic["include_recordings_val"] = SSUP_TRAININGSET
        elif comb_dic["include_recordings_train"] == SSUP_TESTINGSET and comb_dic["include_recordings_val"] == HUI_TRAININGSET:
            comb_dic["cross_eval_type"] = "reverse_testtrain_ssup_train_vs_hui_test"
            comb_dic["include_recordings_train"] = SSUP_TESTINGSET
            comb_dic["include_recordings_val"] = HUI_TRAININGSET
            
            
        # EntranceCBack_vs_SSUPTesting
        # EntranceCBack_and_Cafeteria_vs_SSUPTesting
        # EntranceCBack_and_Cafeteria_and_CoffeeB_vs_SSUPTesting
        # EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_vs_SSUPTesting
        # EntanceCBack_vs_HUITesting
        # EntranceCBack_and_Cafeteria_vs_HUITesting
        # EntranceCBack_and_Cafeteria_and_CoffeeB_vs_HUITesting
        # EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_vs_HUITesting
        # Cafeteria_vs_HUITesting
        # CoffeeB_vs_HUITesting
        # MainEntrance_vs_HUITesting
        elif comb_dic["include_recordings_train"] == "EntranceCBack_vs_SSUPTesting_TRAIN" and comb_dic["include_recordings_val"] == "EntranceCBack_vs_SSUPTesting_VAL":
            comb_dic["cross_eval_type"] = "EntranceCBack_VS_SSUPTesting"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_VS_SSUPTESTING_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_VS_SSUPTESTING_VAL
            # prInfo(f"{valid_comb} EntranceCBack_VS_SSUPTesting")
        elif comb_dic["include_recordings_train"] == "EntranceCBack_and_Cafeteria_vs_SSUPTesting_TRAIN" and comb_dic["include_recordings_val"] == "EntranceCBack_and_Cafeteria_vs_SSUPTesting_VAL":
            comb_dic["cross_eval_type"] = "EntranceCBack_and_Cafeteria_VS_SSUPTesting"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_AND_CAFETERIA_VS_SSUPTESTING_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_AND_CAFETERIA_VS_SSUPTESTING_VAL
            # prInfo(f"{valid_comb} EntranceCBack_and_Cafeteria_VS_SSUPTesting")
        elif comb_dic["include_recordings_train"] == "EntranceCBack_and_Cafeteria_and_CoffeeB_vs_SSUPTesting_TRAIN" and comb_dic["include_recordings_val"] == "EntranceCBack_and_Cafeteria_and_CoffeeB_vs_SSUPTesting_VAL":
            comb_dic["cross_eval_type"] = "EntranceCBack_and_Cafeteria_and_CoffeeB_VS_SSUPTesting"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_SSUPTESTING_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_SSUPTESTING_VAL
            # prInfo(f"{valid_comb} EntranceCBack_and_Cafeteria_and_CoffeeB_VS_SSUPTesting")
        elif comb_dic["include_recordings_train"] == "EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_vs_SSUPTesting_TRAIN" and comb_dic["include_recordings_val"] == "EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_vs_SSUPTesting_VAL":
            comb_dic["cross_eval_type"] = "EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_VS_SSUPTesting"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_SSUPTESTING_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_SSUPTESTING_VAL
            # prInfo(f"{valid_comb} EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_VS_SSUPTesting")
        elif comb_dic["include_recordings_train"] == "EntanceCBack_vs_HUITesting_TRAIN" and comb_dic["include_recordings_val"] == "EntanceCBack_vs_HUITesting_VAL":
            comb_dic["cross_eval_type"] = "EntanceCBack_VS_HUITesting"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_VS_HUITESTING_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_VS_HUITESTING_VAL
            # prInfo(f"{valid_comb} EntanceCBack_VS_HUITesting")
        elif comb_dic["include_recordings_train"] == "EntranceCBack_and_Cafeteria_vs_HUITesting_TRAIN" and comb_dic["include_recordings_val"] == "EntranceCBack_and_Cafeteria_vs_HUITesting_VAL":
            comb_dic["cross_eval_type"] = "EntranceCBack_and_Cafeteria_VS_HUITesting"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_AND_CAFETERIA_VS_HUITESTING_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_AND_CAFETERIA_VS_HUITESTING_VAL
            # prInfo(f"{valid_comb} EntranceCBack_and_Cafeteria_VS_HUITesting")
        elif comb_dic["include_recordings_train"] == "EntranceCBack_and_Cafeteria_and_CoffeeB_vs_HUITesting_TRAIN" and comb_dic["include_recordings_val"] == "EntranceCBack_and_Cafeteria_and_CoffeeB_vs_HUITesting_VAL":
            comb_dic["cross_eval_type"] = "EntranceCBack_and_Cafeteria_and_CoffeeB_VS_HUITesting"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_HUITESTING_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_HUITESTING_VAL
            # prInfo(f"{valid_comb} EntranceCBack_and_Cafeteria_and_CoffeeB_VS_HUITesting")
        elif comb_dic["include_recordings_train"] == "EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_vs_HUITesting_TRAIN" and comb_dic["include_recordings_val"] == "EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_vs_HUITesting_VAL":
            comb_dic["cross_eval_type"] = "EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_VS_HUITesting"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_HUITESTING_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_HUITESTING_VAL
            # prInfo(f"{valid_comb} EntanceCBack_and_Cafeteria_and_CoffeeB_and_MainEntrance_VS_HUITesting")
        elif comb_dic["include_recordings_train"] == "Cafeteria_vs_HUITesting_TRAIN" and comb_dic["include_recordings_val"] == "Cafeteria_vs_HUITesting_VAL":
            comb_dic["cross_eval_type"] = "Cafeteria_VS_HUITesting"
            comb_dic["include_recordings_train"] = CAFETERIA_VS_HUITESTING_TRAIN
            comb_dic["include_recordings_val"] = CAFETERIA_VS_HUITESTING_VAL
            # prInfo(f"{valid_comb} Cafeteria_VS_HUITesting")
        elif comb_dic["include_recordings_train"] == "CoffeeB_vs_HUITesting_TRAIN" and comb_dic["include_recordings_val"] == "CoffeeB_vs_HUITesting_VAL":
            comb_dic["cross_eval_type"] = "CoffeeB_VS_HUITesting"
            comb_dic["include_recordings_train"] = COFFEEB_VS_HUITESTING_TRAIN
            comb_dic["include_recordings_val"] = COFFEEB_VS_HUITESTING_VAL
            # prInfo(f"{valid_comb} CoffeeB_VS_HUITesting")
        elif comb_dic["include_recordings_train"] == "MainEntrance_vs_HUITesting_TRAIN" and comb_dic["include_recordings_val"] == "MainEntrance_vs_HUITesting_VAL":
            comb_dic["cross_eval_type"] = "MainEntrance_VS_HUITesting"
            comb_dic["include_recordings_train"] = MAINENTRANCE_VS_HUITESTING_TRAIN
            comb_dic["include_recordings_val"] = MAINENTRANCE_VS_HUITESTING_VAL
            # prInfo(f"{valid_comb} MainEntrance_VS_HUITesting")
        
        
        elif comb_dic["include_recordings_train"] == "Room104_vs_EntranceCBack_TRAIN" and comb_dic["include_recordings_val"] == "Room104_vs_EntranceCBack_VAL":
            comb_dic["cross_eval_type"] = "Room104_VS_EntranceCBack"
            comb_dic["include_recordings_train"] = ROOM104_VS_ENTRANCECBACK_TRAIN
            comb_dic["include_recordings_val"] = ROOM104_VS_ENTRANCECBACK_VAL
            # prInfo(f"{valid_comb} Room104_VS_EntranceCBack")
        elif comb_dic["include_recordings_train"] == "EntranceCBack_vs_Room104_TRAIN" and comb_dic["include_recordings_val"] == "EntranceCBack_vs_Room104_VAL":
            comb_dic["cross_eval_type"] = "EntranceCBack_VS_Room104"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_VS_ROOM104_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_VS_ROOM104_VAL
            # prInfo(f"{valid_comb} EntranceCBack_VS_Room104")
        elif comb_dic["include_recordings_train"] == "All_VS_EntranceCBack_TRAIN" and comb_dic["include_recordings_val"] == "All_VS_EntranceCBack_VAL":
            comb_dic["cross_eval_type"] = "All_VS_EntranceCBack"
            comb_dic["include_recordings_train"] = ALL_VS_ENTRANCECBACK_TRAIN
            comb_dic["include_recordings_val"] = ALL_VS_ENTRANCECBACK_VAL
            # prInfo(f"{valid_comb} All_VS_EntranceCBack")
        elif comb_dic["include_recordings_train"] == "All_VS_Room104_TRAIN" and comb_dic["include_recordings_val"] == "All_VS_Room104_VAL":
            comb_dic["cross_eval_type"] = "All_VS_Room104"
            comb_dic["include_recordings_train"] = ALL_VS_ROOM104_TRAIN
            comb_dic["include_recordings_val"] = ALL_VS_ROOM104_VAL
            # prInfo(f"{valid_comb} All_VS_Room104")
        elif comb_dic["include_recordings_train"] == "Intra_Room104_With_Setups_TRAIN" and comb_dic["include_recordings_val"] == "Intra_Room104_With_Setups_VAL":
            comb_dic["cross_eval_type"] = "Intra_Room104_With_Setups"
            comb_dic["include_recordings_train"] = INTRA_ROOM104_WITH_SETUPS_TRAIN
            comb_dic["include_recordings_val"] = INTRA_ROOM104_WITH_SETUPS_VAL
            # prInfo(f"{valid_comb} Intra_Room104_With_Setups")
        elif comb_dic["include_recordings_train"] == "Intra_EntranceCBack_With_Setups_TRAIN" and comb_dic["include_recordings_val"] == "Intra_EntranceCBack_With_Setups_VAL":
            comb_dic["cross_eval_type"] = "Intra_EntranceCBack_With_Setups"
            comb_dic["include_recordings_train"] = INTRA_ENTRANCECBACK_WITH_SETUPS_TRAIN
            comb_dic["include_recordings_val"] = INTRA_ENTRANCECBACK_WITH_SETUPS_VAL
            # prInfo(f"{valid_comb} Intra_EntranceCBack_With_Setups")
        elif comb_dic["include_recordings_train"] == "Cafeteria_vs_Room104_TRAIN" and comb_dic["include_recordings_val"] == "Cafeteria_vs_Room104_VAL":
            comb_dic["cross_eval_type"] = "Cafeteria_VS_Room104"
            comb_dic["include_recordings_train"] = CAFETERIA_VS_ROOM104_TRAIN
            comb_dic["include_recordings_val"] = CAFETERIA_VS_ROOM104_VAL
            # prInfo(f"{valid_comb} Cafeteria_VS_Room104")
        elif comb_dic["include_recordings_train"] == "CoffeeB_vs_Room104_TRAIN" and comb_dic["include_recordings_val"] == "CoffeeB_vs_Room104_VAL":
            comb_dic["cross_eval_type"] = "CoffeeB_VS_Room104"
            comb_dic["include_recordings_train"] = COFFEEB_VS_ROOM104_TRAIN
            comb_dic["include_recordings_val"] = COFFEEB_VS_ROOM104_VAL
            # prInfo(f"{valid_comb} CoffeeB_VS_Room104")
        elif comb_dic["include_recordings_train"] == "MainEntrance_vs_Room104_TRAIN" and comb_dic["include_recordings_val"] == "MainEntrance_vs_Room104_VAL":
            comb_dic["cross_eval_type"] = "MainEntrance_VS_Room104"
            comb_dic["include_recordings_train"] = MAINENTRANCE_VS_ROOM104_TRAIN
            comb_dic["include_recordings_val"] = MAINENTRANCE_VS_ROOM104_VAL
            # prInfo(f"{valid_comb} MainEntrance_VS_Room104")
        elif comb_dic["include_recordings_train"] == "EntranceCBack_and_Cafetaria_vs_Room104_TRAIN" and comb_dic["include_recordings_val"] == "EntranceCBack_and_Cafetaria_vs_Room104_VAL":
            comb_dic["cross_eval_type"] = "EntranceCBack_and_Cafetaria_VS_Room104"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_AND_CAFETERIA_VS_ROOM104_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_AND_CAFETERIA_VS_ROOM104_VAL
            # prInfo(f"{valid_comb} EntranceCBack_and_Cafetaria_VS_Room104")
        elif comb_dic["include_recordings_train"] == "EntranceCBack_and_Cafetaria_and_CoffeeB_vs_Room104_TRAIN" and comb_dic["include_recordings_val"] == "EntranceCBack_and_Cafetaria_and_CoffeeB_vs_Room104_VAL":
            comb_dic["cross_eval_type"] = "EntranceCBack_and_Cafetaria_and_CoffeeB_VS_Room104"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_ROOM104_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_ROOM104_VAL
            # prInfo(f"{valid_comb} EntranceCBack_and_Cafetaria_and_CoffeeB_VS_Room104")
        elif comb_dic["include_recordings_train"] == "EntranceCBack_and_Cafetaria_and_CoffeeB_and_MainEntrance_vs_Room104_TRAIN" and comb_dic["include_recordings_val"] == "EntranceCBack_and_Cafetaria_and_CoffeeB_and_MainEntrance_vs_Room104_VAL":
            comb_dic["cross_eval_type"] = "EntranceCBack_and_Cafetaria_and_CoffeeB_and_MainEntrance_VS_Room104"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_ROOM104_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_ROOM104_VAL
            # prInfo(f"{valid_comb} EntranceCBack_and_Cafetaria_and_CoffeeB_and_MainEntrance_VS_Room104")
        elif comb_dic["include_recordings_train"] == "EntranceCBack_vs_AlbeeSquare_TRAIN" and comb_dic["include_recordings_val"] == "EntranceCBack_vs_AlbeeSquare_VAL":
            comb_dic["cross_eval_type"] = "EntranceCBack_VS_AlbeeSquare"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_VS_ALBEESQUARE_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_VS_ALBEESQUARE_VAL
            # prInfo(f"{valid_comb} EntranceCBack_VS_AlbeeSquare")
        elif comb_dic["include_recordings_train"] == "EntranceCBack_and_Cafetaria_vs_AlbeeSquare_TRAIN" and comb_dic["include_recordings_val"] == "EntranceCBack_and_Cafetaria_vs_AlbeeSquare_VAL":
            comb_dic["cross_eval_type"] = "EntranceCBack_and_Cafetaria_VS_AlbeeSquare"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_AND_CAFETERIA_VS_ALBEESQUARE_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_AND_CAFETERIA_VS_ALBEESQUARE_VAL
            # prInfo(f"{valid_comb} EntranceCBack_and_Cafetaria_VS_AlbeeSquare")
        elif comb_dic["include_recordings_train"] == "EntranceCBack_and_Cafetaria_and_CoffeeB_vs_AlbeeSquare_TRAIN" and comb_dic["include_recordings_val"] == "EntranceCBack_and_Cafetaria_and_CoffeeB_vs_AlbeeSquare_VAL":
            comb_dic["cross_eval_type"] = "EntranceCBack_and_Cafetaria_and_CoffeeB_VS_AlbeeSquare"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_ALBEESQUARE_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_VS_ALBEESQUARE_VAL
            # prInfo(f"{valid_comb} EntranceCBack_and_Cafetaria_and_CoffeeB_VS_AlbeeSquare")
        elif comb_dic["include_recordings_train"] == "EntranceCBack_and_Cafetaria_and_CoffeeB_and_MainEntrance_vs_AlbeeSquare_TRAIN" and comb_dic["include_recordings_val"] == "EntranceCBack_and_Cafetaria_and_CoffeeB_and_MainEntrance_vs_AlbeeSquare_VAL":
            comb_dic["cross_eval_type"] = "EntranceCBack_and_Cafetaria_and_CoffeeB_and_MainEntrance_VS_AlbeeSquare"
            comb_dic["include_recordings_train"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_ALBEESQUARE_TRAIN
            comb_dic["include_recordings_val"] = ENTRANCECBACK_AND_CAFETERIA_AND_COFFEEB_AND_MAINENTRANCE_VS_ALBEESQUARE_VAL
            # prInfo(f"{valid_comb} EntranceCBack_and_Cafetaria_and_CoffeeB_and_MainEntrance_VS_AlbeeSquare")
        elif comb_dic["include_recordings_train"] == "HUI_vs_AstorPlace_TRAIN" and comb_dic["include_recordings_val"] == "HUI_vs_AstorPlace_VAL":
            comb_dic["cross_eval_type"] = "HUI_VS_AstorPlace"
            comb_dic["include_recordings_train"] = HUI_VS_ASTORPLACE_TRAIN
            comb_dic["include_recordings_val"] = HUI_VS_ASTORPLACE_VAL
            # prInfo(f"{valid_comb} HUI_VS_AstorPlace")
        elif comb_dic["include_recordings_train"] == "HUI_vs_AlbeeSquare_TRAIN" and comb_dic["include_recordings_val"] == "HUI_vs_AlbeeSquare_VAL":
            comb_dic["cross_eval_type"] = "HUI_VS_AlbeeSquare"
            comb_dic["include_recordings_train"] = HUI_VS_ALBEESQUARE_TRAIN
            comb_dic["include_recordings_val"] = HUI_VS_ALBEESQUARE_VAL
            # prInfo(f"{valid_comb} HUI_VS_AlbeeSquare")
        elif comb_dic["include_recordings_train"] == "AstorPlace_vs_AlbeeSquare_TRAIN" and comb_dic["include_recordings_val"] == "AstorPlace_vs_AlbeeSquare_VAL":
            comb_dic["cross_eval_type"] = "AstorPlace_VS_AlbeeSquare"
            comb_dic["include_recordings_train"] = ASTORPLACE_VS_ALBEESQUARE_TRAIN
            comb_dic["include_recordings_val"] = ASTORPLACE_VS_ALBEESQUARE_VAL
            # prInfo(f"{valid_comb} AstorPlace_VS_AlbeeSquare")
        elif comb_dic["include_recordings_train"] == "HUI_and_AstorPlace_vs_AlbeeSquare_TRAIN" and comb_dic["include_recordings_val"] == "HUI_and_AstorPlace_vs_AlbeeSquare_VAL":
            comb_dic["cross_eval_type"] = "HUI_and_AstorPlace_VS_AlbeeSquare"
            comb_dic["include_recordings_train"] = HUI_AND_ASTORPLACE_VS_ALBEESQUARE_TRAIN
            comb_dic["include_recordings_val"] = HUI_AND_ASTORPLACE_VS_ALBEESQUARE_VAL
            # prInfo(f"{valid_comb} HUI_and_AstorPlace_VS_AlbeeSquare")
        else:
            valid_comb = False

        
        comb_dic["valid"] = valid_comb
            
        all_dics[combcount] = comb_dic

    valid_combinations =  [d for d in all_dics.values() if d["valid"]]
    prInfo("Got {} valid combinations of parameters".format(len(valid_combinations)))

    counter = 0
    for _, comb_dic in all_dics.items():
        if comb_dic["valid"]:
            
            comb_dic["hostname"] = hostname
            
            if comb_dic["train_tracks_filename"].endswith("fold"):
                
                train_tracks_filename_base = comb_dic["train_tracks_filename"]
                val_tracks_filename_base = comb_dic["val_tracks_filename"]
                test_tracks_filename_base = comb_dic["test_tracks_filename"]
                for fold in range(5):
                    comb_dic["experiment_name"] = f"expe_{next_expe_id:03d}_{counter:05d}_fold_{fold}"
                    comb_dic["experiment_group"] = f"expe_{next_expe_id:03d}"
                    comb_dic["experiment_iter"] = counter
                    comb_dic["fold"] = fold
                    comb_dic["train_tracks_filename"] = train_tracks_filename_base.replace("fold", f"fold_{fold}.txt")
                    comb_dic["val_tracks_filename"] = val_tracks_filename_base.replace("fold", f"fold_{fold}.txt")
                    comb_dic["test_tracks_filename"] = test_tracks_filename_base.replace("fold", f"fold_{fold}.txt")
                    file_out = os.path.join(expe_path, "{:05d}_fold_{:02d}.yaml".format(int(counter), fold))
                    
                    write_dic_to_yaml_file(comb_dic, file_out)
                                    
                counter += 1
                
            else:

                comb_dic["experiment_name"] = f"expe_{next_expe_id:03d}_{counter:05d}"
                file_out = os.path.join(expe_path, "{:05d}.yaml".format(int(counter)))
                                
                counter += 1
                write_dic_to_yaml_file(comb_dic, file_out)


                TEMPLATE = 'oarsub -p "(gpu_compute_capability == 7.0 OR gpu_compute_capability == 7.2 OR gpu_compute_capability == 7.5 OR gpu_compute_capability == 8.0 OR gpu_compute_capability == 8.6 OR gpu_compute_capability == 8.9 OR gpu_compute_capability == 9.0) AND gpu_mem>=8000" -l host=1/core=8,gpu=1,walltime=4 "module load conda; conda activate huienv; cd ./public/Projects/github/HUI360-Baselines/; python training.py -hp ./experiments/generated_configs/expe_EXPEID -uw -sm -pd -pn MYPROJECTNAME -eif FROMIDX -eit TOIDX -nw 8" -q besteffort'
                
                command_line = TEMPLATE.replace("EXPEID", f"{next_expe_id:03d}")
                command_line = command_line.replace("FROMIDX", f"{counter-1}")
                command_line = command_line.replace("TOIDX", f"{counter}")
                command_line = command_line.replace("MYPROJECTNAME", f"{args.project_name}")

                if args.no_best_effort:
                    command_line = command_line.replace(" -q besteffort", "")
                    
                if args.gpu_mem != "":
                    command_line = command_line.replace("gpu_mem>=8000", f"gpu_mem>={args.gpu_mem}")
                elif ("mb_" in comb_dic["force_model_type"] and ("finetune" in comb_dic["force_model_type"] or "scratch" in comb_dic["force_model_type"])) and not ("lite" in comb_dic["force_model_type"]):
                    command_line = command_line.replace("gpu_mem>=8000", "gpu_mem>=40000")
                elif ("mb_" in comb_dic["force_model_type"] and ("finetune" in comb_dic["force_model_type"] or "scratch" in comb_dic["force_model_type"])) and ("lite" in comb_dic["force_model_type"]):
                    command_line = command_line.replace("gpu_mem>=8000", "gpu_mem>=32000")
                    
                print(command_line)
                
                # print(comb_dic["cross_eval_type"])
    
    # copy the current file for reference
    shutil.copy(os.path.join(here, "create_config_files.py"), os.path.join(expe_path, "create_config_files.py.bk"))

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", "-pn", type=str, default="tests_", help="Wandb project name")
    parser.add_argument("--no_best_effort", "-nbe", action="store_true", help="Do not submit the best effort job")
    parser.add_argument("--gpu_mem", "-gm", type=str, default="", help="Minimum GPU memory required")
    args = parser.parse_args()
    main(args)
