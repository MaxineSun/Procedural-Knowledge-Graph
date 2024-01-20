#!/bin/bash
#SBATCH --job-name=ori_CR_5_gpt2-xl_16_5
#SBATCH --output=ori_CR_5_gpt2-xl_16_5.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100-pcie-40gb:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 5 --dataset "CR" --parallel_id 5 --model 'gpt2-xl' --shuffle_mode 'ori'