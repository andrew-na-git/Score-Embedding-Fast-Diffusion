#!/bin/bash
#SBATCH --gpus=2
#SBATCH -t 8:0:0
#SBATCH --mem=32G
source ../.venv/bin/activate

python generate.py