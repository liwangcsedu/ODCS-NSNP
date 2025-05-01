#!/bin/bash
# Run the main.py in the background

nohup python train.py > trainlog.txt 2>&1 &