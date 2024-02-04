#!/bin/bash
#SBATCH --job-name=sorted_2_gpt-j-6b_16_5
#SBATCH --output=sorted_2_gpt-j-6b_16_5.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --random_seed 24 --sequence_length 16 --data_classes 2 --dataset "sst2" --parallel_id 5 --model 'gpt-j-6b' --shuffle_mode 'sorted'