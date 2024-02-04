#!/bin/bash
#SBATCH --job-name=ori_14_gpt-j-6b_16
#SBATCH --output=ori_14_gpt-j-6b_16.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_overall.py --sequence_length 16 --data_classes 14 --dataset "dbpedia_14" --model 'gpt-j-6b' --shuffle_mode 'ori' --random_seed 69