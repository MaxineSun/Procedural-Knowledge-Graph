#!/bin/bash
#SBATCH --job-name=bubble_14_gpt-j-6b_16_0
#SBATCH --output=bubble_14_gpt-j-6b_16_0.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 14 --dataset "dbpedia_14" --parallel_id 0 --model 'gpt-j-6b' --shuffle_mode 'bubble'