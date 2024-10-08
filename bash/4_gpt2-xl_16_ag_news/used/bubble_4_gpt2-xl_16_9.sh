#!/bin/bash
#SBATCH --job-name=bubble_4_gpt2-xl_16_9
#SBATCH --output=bubble_4_gpt2-xl_16_9.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=titan_rtx:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --random_seed 42 --sequence_length 16 --data_classes 4 --dataset "ag_news" --parallel_id 9 --model 'gpt2-xl' --shuffle_mode 'bubble'