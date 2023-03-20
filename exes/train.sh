#!/bin/bash

python train.py --version='8' --size='s' --data='data/timelapse.yaml' --img_size 3840 2160 --batch_size=2 --epochs=10 --save_period=1 --device=0