#!/bin/bash

# Slurm sbatch options
#SBATCH -n 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o run_main_webqsp.sh.log
#export DISKCACHE_DIRECTORY=/state/partition1/user/$USER/outlines_cache

mkdir -p $TMPDIR/.outlines
export OUTLINES_CACHE_DIR=$TMPDIR/.outlines
export CUDA_VISIBLE_DEVICES=0

export WANDB_API_KEY="b1cf013fa15f74b678f33b1c935f969a8fef57ae"
WANDB_MODE=offline

module load anaconda/2023a-pytorch
module load nccl
source activate reasoner

echo "starting experiment"

python main.py -d webqsp --prompt_mode scored_100 -p /home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/webqsp_Jan03-19:20:34/retrieval_result.pth -m /home/gridsan/mhadjiivanov/meng/SubgraphRAG/hf/models/Llama-3.2-1B-Instruct

#python main.py -d webqsp --prompt_mode scored_200 -p /home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/webqsp_Jan03-19:20:34/retrieval_result.pth

#python main.py -d webqsp --prompt_mode scored_300 -p /home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/webqsp_Jan03-19:20:34/retrieval_result.pth

#python main.py -d cwq --prompt_mode scored_100 -p /home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/cwq_Jan04-01:33:49/retrieval_result.pth

#python main.py -d cwq --prompt_mode scored_200 -p /home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/cwq_Jan04-01:33:49/retrieval_result.pth

#python main.py -d cwq --prompt_mode scored_300 -p /home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/cwq_Jan04-01:33:49/retrieval_result.pth

echo "done"