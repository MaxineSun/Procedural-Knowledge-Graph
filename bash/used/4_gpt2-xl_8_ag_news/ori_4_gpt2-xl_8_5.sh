#!/bin/bash
#SBATCH --job-name=ori_4_gpt2-xl_8_5
#SBATCH --output=ori_4_gpt2-xl_8_5.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=titan_rtx:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 8 --data_classes 4 --dataset "ag_news" --parallel_id 5 --model 'gpt2-xl' --shuffle_mode 'ori'