#!/bin/bash
#SBATCH --job-name=sorted_5_gpt-j-6b_16_21
#SBATCH --output=sorted_5_gpt-j-6b_16_21.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --random_seed 42 --sequence_length 16 --data_classes 5 --dataset "sst5" --parallel_id 21 --model 'gpt-j-6b' --shuffle_mode 'sorted'