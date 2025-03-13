#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v5
#SBATCH --gpus-per-node=1
#SBATCH --time=88:00:00
#SBATCH --partition=hard
#SBATCH --constraint=A6000
#SBATCH --output=train/results/run.out


export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"


unset OLLAMA_DEBUG
unset OLLAMA_LOG_LEVEL
unset OLLAMA_VERBOSE

python ../run.py \
  --output_dir ../models/T2_M5_CPE_B1_E0/ALL \
  --curriculum_model llama3.2 \
  --num_iterations 102
