import os 
import sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join("..", here))

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def get_best_threshold_f1(all_labels, all_probabilities, thresholds_min_max=False):
    """Get the best threshold for the model.
    Args:
        all_labels: list of labels
        all_probabilities: list of probabilities
        thresholds_min_max: if True, the thresholds will be the minimum and maximum probabilities
    Returns:
        best_threshold: the best threshold
        best_f1: the best f1 score
    """
    if thresholds_min_max:
        thresholds = np.linspace(min(all_probabilities), max(all_probabilities), 100)
    else:
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
