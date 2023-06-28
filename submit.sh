#!/bin/bash
#SBATCH --job-name=entity_resolution
#SBATCH --output=er_out.out
#SBATCH --partition=my_partition
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1

python main.py
