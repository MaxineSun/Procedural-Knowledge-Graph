#!/bin/bash
#SBATCH --job-name=ori_4_gpt2-xl_16_10
#SBATCH --output=ori_4_gpt2-xl_16_10.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=titan_rtx:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --random_seed 42 --sequence_length 16 --data_classes 4 --dataset "ag_news" --parallel_id 10 --model 'gpt2-xl' --shuffle_mode 'ori'