#!/bin/bash
#SBATCH --job-name=ori_2_gpt-j-6b_16_4
#SBATCH --output=ori_2_gpt-j-6b_16_4.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --random_seed 12 --sequence_length 16 --data_classes 2 --dataset "sst2" --parallel_id 4 --model 'gpt-j-6b' --shuffle_mode 'ori'