#!/bin/bash

cd ~/WindNODE || exit 1

export PYTHONPATH=src

mkdir -p logs

TS=$(date +"%d-%m-%Y_%I-%M%p")

nohup python scripts/node_impl.py \
    --data-path data/T1.csv \
    --output-dir outputs/node \
    --epochs 10000 \
    > logs/node_train_${TS}.log 2>&1 &

echo "NODE training started in background."
echo "Log file: logs/node_train_${TS}.log"
