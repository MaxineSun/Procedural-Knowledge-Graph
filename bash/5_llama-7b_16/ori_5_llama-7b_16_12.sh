#!/bin/bash
#SBATCH --job-name=ori_5_Llama-2-7b-hf_16_12
#SBATCH --output=ori_5_Llama-2-7b-hf_16_12.out 
#SBATCH --mem-per-cpu=60G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=36:00:00
# module load eth_proxy

python3 ../../openicl_sort.py --sequence_length 16 --data_classes 5 --dataset "sst5" --parallel_id 12 --model 'NousResearch/Llama-2-7b-hf' --shuffle_mode 'ori'