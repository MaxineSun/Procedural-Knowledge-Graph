#!/bin/bash
#SBATCH --job-name=ori_CR_2_gpt-j-6b_8_3
#SBATCH --output=ori_CR_2_gpt-j-6b_8_3.out 
#SBATCH --mem-per-cpu=63G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=36:33:33
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 8 --data_classes 2 --dataset "CR" --parallel_id 3 --model 'gpt-j-6b' --shuffle_mode 'ori'