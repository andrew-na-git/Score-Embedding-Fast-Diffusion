#!/bin/bash

#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH -t 12:00:00

source ../.venv/bin/activate

python run.py