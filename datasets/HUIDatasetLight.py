from torch.utils.data import Dataset
import torch
import pickle
import os

class HUIInteract360Light(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        
        light_dataset_pickle = pickle.load(open(dataset_path, "rb"))
        self.inputs = light_dataset_pickle["inputs"]
        self.labels = light_dataset_pickle["labels"]
        self.metadata = light_dataset_pickle["metadata"]
        
        self.total_positives = sum(self.labels)
        self.total_negatives = len(self.labels) - self.total_positives
        self.total_samples = len(self.labels)
        self.data_columns_in_dataset = ['mask_size', 'xmin', 'ymin', 'xmax', 'ymax', 'vitpose_nose_x', 
                                        'vitpose_nose_y', 'vitpose_nose_score', 'vitpose_left_eye_x', 'vitpose_left_eye_y', 
                                        'vitpose_left_eye_score', 'vitpose_right_eye_x', 'vitpose_right_eye_y', 'vitpose_right_eye_score', 
                                        'vitpose_left_ear_x', 'vitpose_left_ear_y', 'vitpose_left_ear_score', 'vitpose_right_ear_x', 'vitpose_right_ear_y', 
                                        'vitpose_right_ear_score', 'vitpose_left_shoulder_x', 'vitpose_left_shoulder_y', 'vitpose_left_shoulder_score', 'vitpose_right_shoulder_x', 
                                        'vitpose_right_shoulder_y', 'vitpose_right_shoulder_score', 'vitpose_left_elbow_x', 'vitpose_left_elbow_y', 'vitpose_left_elbow_score', 
                                        'vitpose_right_elbow_x', 'vitpose_right_elbow_y', 'vitpose_right_elbow_score', 'vitpose_left_wrist_x', 'vitpose_left_wrist_y', 'vitpose_left_wrist_score', 
                                        'vitpose_right_wrist_x', 'vitpose_right_wrist_y', 'vitpose_right_wrist_score', 'vitpose_left_hip_x', 'vitpose_left_hip_y', 'vitpose_left_hip_score', 
                                        'vitpose_right_hip_x', 'vitpose_right_hip_y', 'vitpose_right_hip_score', 'vitpose_left_knee_x', 'vitpose_left_knee_y', 'vitpose_left_knee_score', 'vitpose_right_knee_x', 
                                        'vitpose_right_knee_y', 'vitpose_right_knee_score', 'vitpose_left_ankle_x', 'vitpose_left_ankle_y', 'vitpose_left_ankle_score', 'vitpose_right_ankle_x', 
                                        'vitpose_right_ankle_y', 'vitpose_right_ankle_score'] # all the D3 columns hardcoded
        self.input_length_in_frames = self.inputs[0].shape[0]
        self.subsample_frames = 1

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx]), self.metadata[idx]