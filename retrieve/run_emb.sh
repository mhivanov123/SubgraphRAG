#!/bin/bash

# Slurm sbatch options
#SBATCH -n 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o myScript.sh.log-%j

module load anaconda/2023a-pytorch
source activate gte_large_en_v1-5

echo "starting experiment"

python emb.py -d metaqa

echo "done"