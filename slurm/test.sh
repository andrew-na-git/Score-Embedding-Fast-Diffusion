#!/bin/bash

python run.py --sigma=2 \
    --epochs=300 \
    --n=5 \
    --res=64 \
    --seed=2 \
    --dataset=imagenet \
    --folder-name=inet5 \
    --step-size=1 \
    --n-timestep=10 \
    --model=OPENAI
