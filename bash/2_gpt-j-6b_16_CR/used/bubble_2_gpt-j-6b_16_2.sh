#!/bin/bash
#SBATCH --job-name=bubble_CR_2_gpt-j-6b_16_2
#SBATCH --output=bubble_CR_2_gpt-j-6b_16_2.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 2 --dataset "CR" --parallel_id 2 --model 'gpt-j-6b' --shuffle_mode 'bubble'