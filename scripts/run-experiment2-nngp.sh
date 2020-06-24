#nngp_depth=(1 3 5 7)

#nngp_varw=(1.28 2.21 3.14 4.07 5.00)
#nngp_varb=(0.00 0.47 0.93 1.40 1.86)

nngp_depth=(1)
nngp_varw=(1.28)
nngp_varb=(0.00)


pushd ..

mkdir -p /tmp/nngp/output

for nnd in "${nngp_depth[@]}"
do
    for varw in "${nngp_varw[@]}"
    do
        for varb in "${nngp_varb[@]}"
        do
            hpar="nonlinearity=relu,depth=${nnd},weight_var=${varw},bias_var=${varb}"
            echo "TRAIN NNGP with params: ${hpar}"
            outfile="/tmp/gml/output/nngp_${nnd}_${varw}_${varb}.out"
            time python run_experiments.py \
                --num_train=4500 \
                --num_eval=8000 \
                --hparams=${hpar} \
                --n_gauss=501 \
                --n_var=501 \
                --n_corr=500 \
                --max_gauss=10 \
                --use_precomputed_grid=False \
                --dataset='stl10' 2>&1 | tee ${outfile}

            sleep 30
        done
    done
done

popd
