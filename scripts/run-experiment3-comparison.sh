#! /bin/bash

# replace by nngp file
nngp='test_cifar_10000_1.6_1.07.npy'

declare -a nns=(
'nn_cifar10_lr0.001_batch256_depth5_width5_size10000_w1.6_b1.07_predictions.npy'
'nn_cifar10_lr0.001_batch256_depth5_width50_size10000_w1.6_b1.07_predictions.npy'
'nn_cifar10_lr0.001_batch256_depth5_width500_size10000_w1.6_b1.07_predictions.npy'
'nn_cifar10_lr0.001_batch256_depth5_width5000_size10000_w1.6_b1.07_predictions.npy'
)

pushd ..

for nn in "${nns[@]}"
do
    python compare_outputs.py --fnngp ${nngp} --fnn ${nn} --fout "./experiments/ex3_comparison.csv"

done

popd

