#!/bin/bash
#SBATCH --job-name=sorted_CR_2_gpt-j-6b_8_5
#SBATCH --output=sorted_CR_2_gpt-j-6b_8_5.out 
#SBATCH --mem-per-cpu=65G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=36:55:55
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 8 --data_classes 2 --dataset "CR" --parallel_id 5 --model 'gpt-j-6b' --shuffle_mode 'sorted'