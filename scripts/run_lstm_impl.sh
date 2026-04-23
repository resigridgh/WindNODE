#!/bin/bash

cd ~/WindNODE || exit 1

export PYTHONPATH=src

mkdir -p logs

TS=$(date +"%d-%m-%Y_%I-%M%p")

nohup python scripts/lstm_impl.py \
    --data-path data/T1.csv \
    --output-dir outputs/lstm \
    --epochs 10000 \
    --seq-len 12 \
    > logs/lstm_train_${TS}.log 2>&1 &

PID=$!

echo "LSTM training started."
echo "PID: $PID"
echo "Log: logs/lstm_train_${TS}.log"

