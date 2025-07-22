# Vox-MMSD: Voxel-wise Multi-scale and Multi-modal Self-Distillation for Self-supervised Brain Tumor Segmentation
---
This is the souce code for Vox-MMSD: Voxel-wise Multi-scale and Multi-modal Self-Distillation for Self-supervised Brain Tumor Segmentation

## Overall Framework
![](pictures/pipeline.pdf)

## Dataset
Download the BraTS-GLI dataset from BraTS 2023, and put them in the ./BraTS-GLI/source_data/, use
`python ./BraTS-GLI/create_dataset_csv.py`
to preprocess the data and get .csv file for training

## How to use
1. Move into the Pymic-dev and install
```
    cd PyMIC-dev
    pip install -e .
```
2. Move back to the Vox-MMSD dir and run training command
```
    cd ..
    pymic_train ./BraTS-GLI/config/unet3d_voxmmsd.cfg
```
