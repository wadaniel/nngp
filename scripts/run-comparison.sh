#! /bin/bash

nngp='/tmp/nngp/validation_cifar_100_2.97_0.28.npy'
nn='./experiments/nn_cifar10_lr0.01_batch64_depth5_width5_size10000_w1.6_b1.07.npy'
pushd ..

python compare_outputs.py --fnngp $nngp --fnn $nn

popd

