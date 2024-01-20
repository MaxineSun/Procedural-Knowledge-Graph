#!/bin/bash
#SBATCH --job-name=sorted_5_gpt-j-6b_16_18
#SBATCH --output=sorted_5_gpt-j-6b_16_18.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 5 --dataset "sst5" --parallel_id 18 --model 'gpt-j-6b' --shuffle_mode 'sorted'