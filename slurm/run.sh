#!/bin/bash

#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH -t 5:00:00
#SBATCH -w watgpu208

source ../.venv/bin/activate

#cifar 1
# python run.py --sigma=2 \
#         --epochs=500 \
#         --n=1 \
#         --res=32 \
#         --dataset=cifar \
#         --seed=9 \
#         --time-btw-samples=2 \
#         --folder-name=cifar1 \
#         --step-size=1 \
#         --n-timestep=5 \
#         --comparison

# #cifar 3
# python run.py --sigma=2 \
#     --epochs=500 \
#     --n=3 \
#     --res=32 \
#     --dataset=cifar \
#     --seed=56 \
#     --time-btw-samples=2 \
#     --folder-name=cifar3 \
#     --n-timestep=5 \
#     --comparison

# # celeb 1
# python run.py --sigma=2 \
#     --epochs=500 \
#     --n=1 \
#     --res=64 \
#     --dataset=celeb \
#     --time-btw-samples=2 \
#     --folder-name=celeb1 \
#     --n-timestep=10 \
#     --comparison

#celeb 3
python run.py --sigma=2 \
    --epochs=1000 \
    --n=3 \
    --res=64 \
    --dataset=celeb \
    --time-btw-samples=2 \
    --folder-name=celeb3 \
    --step-size=1 \
    --n-timestep=10 \
    --comparison
