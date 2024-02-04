#!/bin/bash
#SBATCH --job-name=bubble_2_Llama-2-7b-hf_8_0
#SBATCH --output=bubble_2_Llama-2-7b-hf_8_0.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=23:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 8 --data_classes 2 --dataset "sst2" --parallel_id 0 --model 'NousResearch/Llama-2-7b-hf' --shuffle_mode 'bubble'