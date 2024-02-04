#!/bin/bash
#SBATCH --job-name=bubble_14_gpt2-xl_16_1
#SBATCH --output=bubble_14_gpt2-xl_16_1.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 14 --dataset "dbpedia_14" --parallel_id 1 --model 'gpt2-xl' --shuffle_mode 'bubble'