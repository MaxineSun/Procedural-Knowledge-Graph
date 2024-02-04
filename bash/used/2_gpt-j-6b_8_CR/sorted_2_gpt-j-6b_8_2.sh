#!/bin/bash
#SBATCH --job-name=sorted_CR_2_gpt-j-6b_8_2
#SBATCH --output=sorted_CR_2_gpt-j-6b_8_2.out 
#SBATCH --mem-per-cpu=62G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=36:22:22
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 8 --data_classes 2 --dataset "CR" --parallel_id 2 --model 'gpt-j-6b' --shuffle_mode 'sorted'