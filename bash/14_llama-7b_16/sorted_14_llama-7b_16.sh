#!/bin/bash
#SBATCH --job-name=sorted_14_llama-7b_16
#SBATCH --output=sorted_14_llama-7b_16.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_overall.py --sequence_length 16 --data_classes 14 --dataset "dbpedia_14" --model 'NousResearch/Llama-2-7b-hf' --shuffle_mode 'sorted' --random_seed 19