#!/bin/bash
#SBATCH --job-name=sorted_2_gpt-j-6b_16_3
#SBATCH --output=sorted_2_gpt-j-6b_16_3.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 2 --dataset "sst2" --parallel_id 3 --model 'gpt-j-6b' --shuffle_mode 'sorted'