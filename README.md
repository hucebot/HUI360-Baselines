# HUI360 - Baselines

Code for baselines of human-robot interaction anticipation on HUI360 dataset as presented in "HUI360: A dataset and baselines for Human Robot Interaction Anticipation" (FG2026).

TODO: Add badges and links

## Legacy baselines code
Please refer to the `legacy` branch of this repository for the results presented in the paper. Updates on the data and code have been made and new baselines have been added in this newer version. 

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

The full skeleton dataset (~59GB) will be automatically downloaded using HuggingFace `snapshot_download` and placed in `datasets/hf_data` when running `training.py` or `infer.py`.

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

A few checkpoints are provided you can download more [here](https://huggingface.co/buckets/rlorlou/hui360-baselines-checkpoints)

python training.py -hp ./experiments/configs/in_hui/mb_base.yaml --save_model -pd -uw -pn baselinesall; python training.py -hp ./experiments/configs/in_hui/mb_ft.yaml --save_model -pd -uw -pn baselinesall; 

python training.py -hp ./experiments/configs/in_ssup/mb_base.yaml --save_model -pd -uw -pn baselinesall; 
python training.py -hp ./experiments/configs/in_ssup/mb_ft.yaml --save_model -pd -uw -pn baselinesall; 
python training.py -hp ./experiments/configs/cross_hui_ssup/mb_base.yaml --save_model -pd -uw -pn baselinesall; 
python training.py -hp ./experiments/configs/cross_hui_ssup/mb_ft.yaml --save_model -pd -uw -pn baselinesall; 
python training.py -hp ./experiments/configs/cross_ssup_hui/mb_base.yaml --save_model -pd -uw -pn baselinesall; 
python training.py -hp ./experiments/configs/cross_ssup_hui/mb_ft.yaml --save_model -pd -uw -pn baselinesall;

### Baselines (HUI)
Common to all models :
- 32 Frames Input (~2.1 second)
- Training and Validation cutoffs at 16 frames (~1.1 second)

#### For HUI (in dataset)
- #Validation Tracks : 352 negatives / 71 positives
- #Training Tracks : 1222 negatives / 216 positives

| Name                        | #Params (Trained) | epochs | AUC   | AP    |
|-----------------------------|-------------------|--------|-------|-------|
| ST-GCN                      | 3.07M             | 100    | 0.880 | 0.581 |
| STG-NF                      | 0.07M             | 150    | 0.805 | 0.492 |
| SkateFormer                 | 1.91M             | 300    | 0.780 | 0.361 |
| MLP                         | 0.07M             | 75     | 0.856 | 0.476 |
| LSTM                        | 0.37M             | 75     | 0.861 | 0.486 |


#### For SSUP-A (in dataset)
- #Validation Tracks : 4842 negatives / 149 positives
- #Training Tracks : 6129 negatives / 136 positives

| Name                        | #Params (Trained) | epochs | AUC   | AP    |
|-----------------------------|-------------------|--------|-------|-------|
| ST-GCN                      | 3.07M             | 100    | 0.868 | 0.213 |
| STG-NF                      | 0.07M             | 150    | 0.758 | 0.082 |
| SkateFormer                 | 1.91M             | 300    | 0.826 | 0.175 |
| MLP                         | 0.07M             | 75     | 0.874 | 0.198 |
| LSTM                        | 0.37M             | 75     | 0.897 | 0.230 |


#### For cross dataset evaluation (train on HUI, test on SSUP-A)
- #Validation Tracks : 4842 negatives / 149 positives
- #Training Tracks : 1222 negatives / 216 positives

| Name                        | #Params (Trained) | epochs | AUC   | AP    |
|-----------------------------|-------------------|--------|-------|-------|
| ST-GCN                      | 3.07M             | 100    | 0.688 | 0.056 |
| STG-NF                      | 0.07M             | 150    | 0.573 | 0.043 |
| SkateFormer                 | 1.91M             | 300    | 0.568 | 0.058 |
| MLP                         | 0.07M             | 75     | 0.770 | 0.112 |
| LSTM                        | 0.37M             | 75     | 0.757 | 0.106 |


#### For cross dataset evaluation (train on SSUP-A, test on HUI)
- #Validation Tracks : 352 negatives / 71 positives
- #Training Tracks : 6129 negatives / 136 positives

| Name                        | #Params (Trained) | epochs | AUC   | AP    |
|-----------------------------|-------------------|--------|-------|-------|
| ST-GCN                      | 3.07M             | 100    | 0.749 | 0.432 |
| STG-NF                      | 0.07M             | 150    | 0.634 | 0.327 |
| SkateFormer                 | 1.91M             | 300    | 0.732 | 0.430 |
| MLP                         | 0.07M             | 75     | 0.778 | 0.463 |
| LSTM                        | 0.37M             | 75     | 0.797 | 0.402 |


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