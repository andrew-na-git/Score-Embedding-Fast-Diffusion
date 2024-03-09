#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:1
#SBATCH --mem=32G
#SBATCH -t 3:00:00

source ../.venv/bin/activate

### DDPM

#cifar 1
python run.py --sigma=2 \
        --epochs=500 \
        --n=1 \
        --res=32 \
        --dataset=cifar \
        --seed=9 \
        --time-btw-samples=2 \
        --folder-name=cifar1 \
        --n-timestep=5 \
        --model=DDPM \
        --comparison

#cifar 3
python run.py --sigma=2 \
    --epochs=500 \
    --n=3 \
    --res=32 \
    --dataset=cifar \
    --seed=56 \
    --time-btw-samples=2 \
    --folder-name=cifar3 \
    --n-timestep=5 \
    --model=DDPM \
    --comparison

# celeb 1
python run.py --sigma=2 \
    --epochs=500 \
    --n=1 \
    --res=64 \
    --dataset=celeb \
    --time-btw-samples=2 \
    --folder-name=celeb1 \
    --n-timestep=10 \
    --model=DDPM \
    --comparison

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
    --model=DDPM \
    --comparison


### openai

#cifar 1
python run.py --sigma=2 \
        --epochs=500 \
        --n=1 \
        --res=32 \
        --dataset=cifar \
        --seed=9 \
        --time-btw-samples=2 \
        --folder-name=cifar1 \
        --n-timestep=5 \
        --model=OPENAI \
        --comparison

#cifar 3
python run.py --sigma=2 \
    --epochs=500 \
    --n=3 \
    --res=32 \
    --dataset=cifar \
    --seed=56 \
    --time-btw-samples=2 \
    --folder-name=cifar3 \
    --n-timestep=5 \
    --model=OPENAI \
    --comparison

# celeb 1
python run.py --sigma=2 \
    --epochs=500 \
    --n=1 \
    --res=64 \
    --dataset=celeb \
    --time-btw-samples=2 \
    --folder-name=celeb1 \
    --n-timestep=10 \
    --model=OPENAI \
    --comparison

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
    --model=OPENAI \
    --comparison

# DDIM

#cifar 1
python run.py --sigma=2 \
        --epochs=500 \
        --n=1 \
        --res=32 \
        --dataset=cifar \
        --seed=9 \
        --time-btw-samples=2 \
        --folder-name=cifar1 \
        --n-timestep=5 \
        --model=DDIM \
        --comparison

#cifar 3
python run.py --sigma=2 \
    --epochs=500 \
    --n=3 \
    --res=32 \
    --dataset=cifar \
    --seed=56 \
    --time-btw-samples=2 \
    --folder-name=cifar3 \
    --n-timestep=5 \
    --model=DDIM \
    --comparison

# celeb 1
python run.py --sigma=2 \
    --epochs=500 \
    --n=1 \
    --res=64 \
    --dataset=celeb \
    --time-btw-samples=2 \
    --folder-name=celeb1 \
    --n-timestep=10 \
    --model=DDIM \
    --comparison

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
    --model=DDIM \
    --comparison
