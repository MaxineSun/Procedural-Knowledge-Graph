#!/bin/bash
#SBATCH --job-name=bubble_CR_5_Llama-2-7b-hf_8_5
#SBATCH --output=bubble_CR_5_Llama-2-7b-hf_8_5.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 8 --data_classes 5 --dataset "CR" --parallel_id 5 --model 'NousResearch/Llama-2-7b-hf' --shuffle_mode 'bubble'