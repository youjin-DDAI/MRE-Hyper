# MRE-Hyper

This is official code of **Physics-Informed Neural Operators for Tissue Elasticity Reconstruction**. We provide training code for reproducibility.

## Environment
- **OS**: Unbuntu 20.04.6 LTS
- **Python**: 3.10.0
- **Pytorch**: 2.1.2
- **Pytorch-lightning**: 2.3.0
- **CUDA**: 11.6+ 
- **GPU**:  NVIDIA RTX A6000

Please refer to the environment.yaml for other required libraries.

## Dataset
We used the FEM Box dataset and the FEM Abdomen dataset, which can be downloaded from the following link. We would like to express our gratitude to the BIOQIC research team for making them publicly available.
https://bioqic-apps.charite.de/downloads

## Training 
- For FEM Box \
python train_unified.py --dataset fem_box --noise-ratio 0.0
- For FEM Abdomen\
python train_unified.py --dataset fem_abdomen --noise-ratio 0.0

