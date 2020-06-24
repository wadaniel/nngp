#! /bin/bash

# train 10k
# valid 5k (hard coded)
# test 10k

pushd ..

python run_experiments.py --num_train=10000 --num_eval=10000 --hparams='nonlinearity=relu,depth=10,weight_var=1.28,bias_var=0.00' --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10 --use_precomputed_grid=True --dataset='cifar'


popd

