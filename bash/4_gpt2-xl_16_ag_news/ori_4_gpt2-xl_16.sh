#!/bin/bash
#SBATCH --job-name=ori_4_gpt2-xl_16
#SBATCH --output=ori_4_gpt2-xl_16.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=titan_rtx:1
#SBATCH --time=4:00:00
# module load eth_proxy

python3 ../../openicl_overall.py --random_seed 37 --sequence_length 16 --data_classes 4 --dataset "ag_news" --model 'gpt2-xl' --shuffle_mode 'ori'