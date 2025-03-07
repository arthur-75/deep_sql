#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v5
#SBATCH --gpus-per-node=1
#SBATCH --time=88:00:00
#SBATCH --partition=hard
#SBATCH --constraint=A6000
#SBATCH --output=train/results/run.out

python ../run.py \
  --output_dir ../models/T2_M5_CPE_B1_E0/ALL \
