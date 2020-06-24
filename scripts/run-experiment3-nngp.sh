#! /bin/bash

use_precomputed=False

pushd ..

python run_experiments.py --num_train=10000 --num_eval=5000 --hparams='nonlinearity=relu,depth=5,weight_var=1.60,bias_var=1.07' --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10 --use_precomputed_grid=${use_precomputed} --dataset='cifar'

popd

