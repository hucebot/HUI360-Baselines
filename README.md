# HUI360 - Baselines

Code for baselines of human-robot interaction anticipation on HUI360 dataset as presented in "HUI360: A dataset and baselines for Human Robot Interaction Anticipation" (FG2026).

## Legacy baselines code
Please refer to the `legacy` branch of this repository. Small updates on the data and code have been made. 

## Installation
Main dependencies are PyTorch and OpenCV-Python for visualization.

```
conda create --name huienv python=3.10
conda activate huienv
pip install -r requirements.txt
```

If you additionally want to use the interactive visualizer install PyQt6
```
PyQt6>=6.0.0
```

Hardware requirement are minimal, training and inference can be performed entirely on CPU or exploit GPU with less than 1GB VRAM.

The full skeleton dataset (~28GB) will be automatically downloaded using HuggingFace `snapshot_download` and placed in `datasets/hf_data` when running `training.py` or `infer.py`.

## Training
You can train a classifier using

```
python training.py -hp ./experiments/configs/in_hui/lstm_base.yaml --save_model
```

## Evaluation
You can evaluate the existing checkpoints (or the ones created during training)

```
python infer.py --model_path ./checkpoints/[SPLIT]/[MODELNAME].pth
```

### Baselines (HUI)
Common to all models :
- 32 Frames Input (~2.1 second)
- Training and Validation cutoffs at 16 frames (~1.1 second)

#### For HUI (in dataset)
- #Validation Tracks : 352 negatives / 71 positives
- #Training Tracks : 1222 negatives / 216 positives

| Name                        | #Params (Trained) | epochs | AUC   | AP    |
|-----------------------------|-------------------|--------|-------|-------|
| LSTM                        | 0.37M             | 100    | XXXXX | XXXXX |
| ST-GCN                      | 3.07M             | 100    | XXXXX | XXXXX |
| MLP                         | 0.07M             | 100    | XXXXX | XXXXX |
| SkateFormer                 | 1.91M             | 600    | XXXXX | XXXXX |
| STG-NF                      | 0.07M             | 600    | XXXXX | XXXXX |


#### For SSUP-A (in dataset)
- #Validation Tracks : 4842 negatives / 149 positives
- #Training Tracks : 6129 negatives / 136 positives

| Name                        | #Params (Trained) | epochs | AUC   | AP    |
|-----------------------------|-------------------|--------|-------|-------|
| LSTM                        | 0.37M             | 100    | XXXXX | XXXXX |
| ST-GCN                      | 3.07M             | 100    | XXXXX | XXXXX |
| MLP                         | 0.07M             | 100    | XXXXX | XXXXX |
| SkateFormer                 | 1.91M             | 600    | XXXXX | XXXXX |
| STG-NF                      | 0.07M             | 600    | XXXXX | XXXXX |

#### For cross dataset evaluation (train on HUI, test on SSUP-A)
- #Validation Tracks : 4842 negatives / 149 positives
- #Training Tracks : 1222 negatives / 216 positives

| Name                        | #Params (Trained) | epochs | AUC   | AP    |
|-----------------------------|-------------------|--------|-------|-------|
| LSTM                        | 0.37M             | 100    | XXXXX | XXXXX |
| ST-GCN                      | 3.07M             | 100    | XXXXX | XXXXX |
| MLP                         | 0.07M             | 100    | XXXXX | XXXXX |
| SkateFormer                 | 1.91M             | 600    | XXXXX | XXXXX |
| STG-NF                      | 0.07M             | 600    | XXXXX | XXXXX |

#### For cross dataset evaluation (train on SSUP-A, test on HUI)
- #Validation Tracks : 352 negatives / 71 positives
- #Training Tracks : 6129 negatives / 136 positives

| Name                        | #Params (Trained) | epochs | AUC   | AP    |
|-----------------------------|-------------------|--------|-------|-------|
| LSTM                        | 0.37M             | 100    | XXXXX | XXXXX |
| ST-GCN                      | 3.07M             | 100    | XXXXX | XXXXX |
| MLP                         | 0.07M             | 100    | XXXXX | XXXXX |
| SkateFormer                 | 1.91M             | 600    | XXXXX | XXXXX |
| STG-NF                      | 0.07M             | 600    | XXXXX | XXXXX |


## Visualization
![image info](./illustrations/datasetvisualizer.jpg)

Visualization is possible with `dataset_visualizer.py`.

<details>
<summary>Using the interactive visualizer</summary>
### Instructions for visualization
TODO
</details>

## Acknoledgements
The code for the SkateFormer, STG-NF, ST-GCN baselines were taken from their respective open-source implementation.

TODO Add Links.