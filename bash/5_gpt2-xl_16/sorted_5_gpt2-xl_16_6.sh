#!/bin/bash
#SBATCH --job-name=sorted_5_gpt2-xl_16_6
#SBATCH --output=sorted_5_gpt2-xl_16_6.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100-pcie-40gb:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 5 --dataset "sst5" --parallel_id 6 --model 'gpt2-xl' --shuffle_mode 'sorted'