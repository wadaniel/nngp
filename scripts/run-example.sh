#! /bin/bash

# GP-7-1.28-0.00

data=cifar
#data=mnist

pushd ..

python run_experiments.py --num_train=100 --num_eval=1000 --hparams='nonlinearity=relu,depth=7,weight_var=1.28,bias_var=0.00' --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10 --use_precomputed_grid=False --dataset=$data

popd

