#!/bin/bash

# Slurm sbatch options
#SBATCH -n 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o webqsp_train.sh.log-%j

module load anaconda/2023a-pytorch
source activate retriever

echo "starting experiment"

export WANDB_API_KEY="b1cf013fa15f74b678f33b1c935f969a8fef57ae"
WANDB_MODE=offline

python train.py -d cwq

echo "done"