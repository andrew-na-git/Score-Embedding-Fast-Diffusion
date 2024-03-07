#!/bin/bash

#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH -t 2:00:00

source ../.venv/bin/activate

# #cifar 1
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
#         --model=OPENAI

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
#     --model=OPENAI

# # celeb 1
# python run.py --sigma=2 \
#     --epochs=1000 \
#     --n=1 \
#     --res=64 \
#     --dataset=celeb \
#     --time-btw-samples=2 \
#     --folder-name=celeb1 \
#     --n-timestep=10 \
#     --model=OPENAI

#celeb 3
# python run.py --sigma=2 \
#     --epochs=1000 \
#     --n=3 \
#     --res=64 \
#     --dataset=celeb \
#     --time-btw-samples=2 \
#     --folder-name=celeb3 \
#     --step-size=1 \
#     --n-timestep=10 \
#     --model=OPENAI

# #celeb 128x128
python run.py --sigma=2 \
    --epochs=1200 \
    --n=2 \
    --res=128 \
    --dataset=celeb \
    --time-btw-samples=2 \
    --folder-name=celeb128 \
    --step-size=1 \
    --n-timestep=20 \
    --model=DDIM
