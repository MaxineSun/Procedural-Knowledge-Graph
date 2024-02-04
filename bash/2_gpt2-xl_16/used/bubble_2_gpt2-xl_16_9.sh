#!/bin/bash
#SBATCH --job-name=bubble_2_gpt2-xl_16_9
#SBATCH --output=bubble_2_gpt2-xl_16_9.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=titan_rtx:1
#SBATCH --time=4:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 2 --dataset "sst2" --parallel_id 9 --model 'gpt2-xl' --shuffle_mode 'bubble'