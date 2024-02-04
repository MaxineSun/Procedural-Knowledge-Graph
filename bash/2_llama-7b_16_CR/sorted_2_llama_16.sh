#!/bin/bash
#SBATCH --job-name=sorted_CR_2_Llama-2-7b-hf_16
#SBATCH --output=sorted_CR_2_Llama-2-7b-hf_16.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=4:00:00
# module load eth_proxy

python3 ../../openicl_overall.py --sequence_length 16 --data_classes 2 --dataset "CR" --model 'NousResearch/Llama-2-7b-hf' --shuffle_mode 'sorted' --random_seed 81