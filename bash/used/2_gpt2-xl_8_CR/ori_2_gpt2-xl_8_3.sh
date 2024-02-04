#!/bin/bash
#SBATCH --job-name=ori_CR_2_gpt2-xl_8_3
#SBATCH --output=ori_CR_2_gpt2-xl_8_3.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=titan_rtx:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 8 --data_classes 2 --dataset "CR" --parallel_id 3 --model 'gpt2-xl' --shuffle_mode 'ori'