#! /bin/bash

pushd ..

python run_experiments.py --num_train=100 --num_eval=1000 --hparams='nonlinearity=relu,depth=10,weight_var=1.28,bias_var=0.00' --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10 --use_precomputed_grid=False --dataset='cifar'

python run_experiments.py --num_train=200 --num_eval=1000 --hparams='nonlinearity=relu,depth=10,weight_var=1.28,bias_var=0.00' --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10 --use_precomputed_grid=True --dataset='cifar'

python run_experiments.py --num_train=400 --num_eval=1000 --hparams='nonlinearity=relu,depth=10,weight_var=1.28,bias_var=0.00' --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10 --use_precomputed_grid=True --dataset='cifar'

python run_experiments.py --num_train=800 --num_eval=1000 --hparams='nonlinearity=relu,depth=10,weight_var=1.28,bias_var=0.00' --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10 --use_precomputed_grid=True --dataset='cifar'

python run_experiments.py --num_train=1000 --num_eval=1000 --hparams='nonlinearity=relu,depth=10,weight_var=1.28,bias_var=0.00' --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10 --use_precomputed_grid=True --dataset='cifar'

popd

