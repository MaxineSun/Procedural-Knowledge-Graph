#!/bin/bash
#SBATCH --job-name=ori_ag_4_gpt-j-6b_16_12
#SBATCH --output=ori_ag_4_gpt-j-6b_16_12.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --random_seed 42 --sequence_length 16 --data_classes 4 --dataset "ag_news" --parallel_id 12 --model 'gpt-j-6b' --shuffle_mode 'ori'