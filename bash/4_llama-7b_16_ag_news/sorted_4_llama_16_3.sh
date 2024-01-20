#!/bin/bash
#SBATCH --job-name=sorted_4_Llama-2-7b-hf_16_3
#SBATCH --output=sorted_4_Llama-2-7b-hf_16_3.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 4 --dataset "ag_news" --parallel_id 3 --model 'NousResearch/Llama-2-7b-hf' --shuffle_mode 'sorted'