#!/bin/bash

# Feel free to use only parts of this file for testing...

RADIUS=5

echo "----------"
echo "Compiling..."
srun -p volta-hp make

echo "----------"
echo "Running serial version..."
srun -p volta-hp --gres=gpu:1 ./cuda-blur-stencil serial $RADIUS ../data/lenna.pbm ../data/result-serial.pbm

echo "----------"
echo "Running CUDA version..."
srun -p volta-hp --gres=gpu:1 ./cuda-blur-stencil cuda $RADIUS ../data/lenna.pbm ../data/result-cuda.pbm

echo "----------"
echo "Comaring results..."
if diff ../data/result-serial.pbm ../data/result-cuda.pbm; then
	echo "OK"
fi
