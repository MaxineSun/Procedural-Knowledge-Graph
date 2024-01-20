#!/bin/bash
#SBATCH --job-name=sorted_4_gpt2-xl_16_2
#SBATCH --output=sorted_4_gpt2-xl_16_2.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=titan_rtx:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 4 --dataset "ag_news" --parallel_id 2 --model 'gpt2-xl' --shuffle_mode 'sorted'