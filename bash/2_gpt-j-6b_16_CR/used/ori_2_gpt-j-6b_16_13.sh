#!/bin/bash
#SBATCH --job-name=ori_CR_2_gpt-j-6b_16_13
#SBATCH --output=ori_CR_2_gpt-j-6b_16_13.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 2 --dataset "CR" --parallel_id 13 --model 'gpt-j-6b' --shuffle_mode 'ori'