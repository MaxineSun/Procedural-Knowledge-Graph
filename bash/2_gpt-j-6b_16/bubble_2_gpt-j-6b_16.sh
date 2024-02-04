#!/bin/bash
#SBATCH --job-name=bubble_2_gpt-j-6b_16
#SBATCH --output=bubble_2_gpt-j-6b_16.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=4:00:00
# module load eth_proxy

python3 ../../openicl_overall.py --sequence_length 16 --data_classes 2 --dataset "sst2" --model 'gpt-j-6b' --shuffle_mode 'bubble' --random_seed 69