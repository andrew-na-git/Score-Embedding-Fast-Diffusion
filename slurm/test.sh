#!/bin/bash

python run.py --sigma=2 \
    --epochs=1000 \
    --n=1 \
    --res=128 \
    --seed=9 \
    --dataset=imagenet \
    --folder-name=inetamp \
    --step-size=1 \
    --n-timestep=10 \
    --model=DDPM
