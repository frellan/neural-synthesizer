#!/bin/bash
#SBATCH --gpus-per-node=V100:1
#SBATCH -t 0-10:00:00

module load GCC/10.2.0
module load CUDA/11.1.1
module load OpenMPI/4.0.5
module load torchvision/0.8.2-PyTorch-1.7.1
pip install ax-platform

python morph_train.py \
    --dataset cifar10 \
    --loss xe \
    --activation relu \
    --hidden_objective srs_upper_tri_alignment \
    --n_classes 10 \
    --schedule_lr True \
    --in_channels 3 \
    --augment_data True \
    --batch_size 512 \
    --n_val 10000 \
    --max_trainset_size 50000
