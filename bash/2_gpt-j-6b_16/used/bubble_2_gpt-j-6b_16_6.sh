#!/bin/bash
#SBATCH --job-name=bubble_2_gpt-j-6b_16_6
#SBATCH --output=bubble_2_gpt-j-6b_16_6.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --random_seed 36 --sequence_length 16 --data_classes 2 --dataset "sst2" --parallel_id 6 --model 'gpt-j-6b' --shuffle_mode 'bubble'