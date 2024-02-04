#!/bin/bash
#SBATCH --job-name=sorted_5_gpt2-xl_16_4
#SBATCH --output=sorted_5_gpt2-xl_16_4.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=titan_rtx:1
#SBATCH --time=4:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --random_seed 42 --sequence_length 16 --data_classes 5 --dataset "sst5" --parallel_id 4 --model 'gpt2-xl' --shuffle_mode 'sorted'