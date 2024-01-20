#!/bin/bash
#SBATCH --job-name=bubble_CR_5_gpt-j-6b_16_15
#SBATCH --output=bubble_CR_5_gpt-j-6b_16_15.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 5 --dataset "CR" --parallel_id 15 --model 'gpt-j-6b' --shuffle_mode 'bubble'