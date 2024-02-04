#!/bin/bash
#SBATCH --job-name=ori_CR_2_gpt2-xl_16
#SBATCH --output=ori_CR_2_gpt2-xl_16.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=titan_rtx:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_overall.py --sequence_length 16 --data_classes 2 --dataset "CR" --model 'gpt2-xl' --shuffle_mode 'ori' --random_seed 34