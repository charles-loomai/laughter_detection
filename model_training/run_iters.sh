#!/bin/bash

for iter in {1..10}; do
    echo "current_iteration = $iter"
    python train_CNN.py $iter
done
