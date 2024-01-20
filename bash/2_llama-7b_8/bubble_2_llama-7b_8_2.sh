#!/bin/bash
#SBATCH --job-name=bubble_2_Llama-2-7b-hf_8_2
#SBATCH --output=bubble_2_Llama-2-7b-hf_8_2.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 8 --data_classes 2 --dataset "sst2" --parallel_id 2 --model 'NousResearch/Llama-2-7b-hf' --shuffle_mode 'bubble'