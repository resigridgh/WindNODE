#!/bin/bash

cd ~/WindNODE || exit 1

export PYTHONPATH=src

mkdir -p logs

# Timestamp
TS=$(date +"%d-%m-%Y_%I-%M%p")

# Run in background
nohup python scripts/cnn_impl.py \
    --data-path data/T1.csv \
    --output-dir outputs/cnn \
    --epochs 10000 \
    > logs/cnn_train_${TS}.log 2>&1 &

echo "CNN training started in background."
echo "Log file: logs/cnn_train_${TS}.log"
