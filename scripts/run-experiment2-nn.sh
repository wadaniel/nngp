learning_rate=(0.001)
decay=(0.000001)

nn_depth=(1 3 5 7)

nn_varw=(2.36 3.44 3.44 4.52 5.59)
nn_varb=(0.00 0.52 1.05 1.57 2.09)

pushd ..

mkdir -p experiments
mkdir -p output

for lr in "${learning_rate[@]}"
do

    for d in "${decay[@]}"
    do

        for nnd in "${nn_depth[@]}"
        do

            for nnw in "${nn_width[@]}"
            do


                for varw in "${nn_varw[@]}"
                do


                    for varb in "${nn_varb[@]}"
                    do
                        
                        echo "TRAIN NN with params:" $lr $d $nnd $nnw $varb
                        outfile="./output/${lr}_${d}_${nnd}_${nnw}_${varb}.out"
                        time python run_nn.py --lr $lr \
                                              --decay $d \
                                              --depth $nnd \
                                              --width $nnw \
                                              --weight_var $varw \
                                              --bias_var $varb \
                                              --dataset "stl10" \
                                              --sub_mean \
                                              --train_size 5000 \
                                              2>&1 | tee ${outfile}

                    done

                done

            done


        done

    done

done

popd
