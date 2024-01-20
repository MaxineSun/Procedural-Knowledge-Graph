#!/bin/bash
#SBATCH --job-name=bubble_ag_4_gpt-j-6b_16_0
#SBATCH --output=bubble_ag_4_gpt-j-6b_16_0.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 4 --dataset "ag_news" --parallel_id 0 --model 'gpt-j-6b' --shuffle_mode 'bubble'