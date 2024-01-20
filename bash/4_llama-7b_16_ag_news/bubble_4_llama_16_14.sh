#!/bin/bash
#SBATCH --job-name=bubble_4_Llama-2-7b-hf_16_14
#SBATCH --output=bubble_4_Llama-2-7b-hf_16_14.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 4 --dataset "ag_news" --parallel_id 14 --model 'NousResearch/Llama-2-7b-hf' --shuffle_mode 'bubble'