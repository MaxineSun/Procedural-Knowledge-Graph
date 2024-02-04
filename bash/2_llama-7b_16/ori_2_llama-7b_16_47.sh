#!/bin/bash
#SBATCH --job-name=ori_2_Llama-2-7b-hf_16_47
#SBATCH --output=ori_2_Llama-2-7b-hf_16_47.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=4:00:00
# module load eth_proxy

python3 ../../openicl_overall.py --sequence_length 16 --data_classes 2 --dataset "sst2" --model 'NousResearch/Llama-2-7b-hf' --shuffle_mode 'ori' --random_seed 47