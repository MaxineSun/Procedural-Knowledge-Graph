#!/bin/bash
#SBATCH --job-name=ori_CR_2_gpt2-xl_16_11
#SBATCH --output=ori_CR_2_gpt2-xl_16_11.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=titan_rtx:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 2 --dataset "CR" --parallel_id 11 --model 'gpt2-xl' --shuffle_mode 'ori'