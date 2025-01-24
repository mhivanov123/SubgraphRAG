#!/bin/bash

# Slurm sbatch options
#SBATCH -n 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o run_emb_webqsp.sh.log

module load anaconda/2023a-pytorch
module load nccl
source activate gte_large_en_v1-5

echo "starting experiment"

python emb.py -d webqsp

echo "done"