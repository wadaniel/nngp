import numpy as np
from load_dataset import load_cifar10

from os import listdir

if __name__ == '__main__':

    DIRECTORY='/tmp/nngp/'
    files = [f for f in listdir(DIRECTORY)]
    (_, _, _, _, _, test_label) = load_cifar10(10000, mean_subtraction=True)
    
    max_acc = 0
    max_acc_f = ''

    min_mse = 1e12
    min_mse_f = ''
    for f in files:
        if 'test_cifar_10000_' in f:
            res = np.load(DIRECTORY+f)
            accuracy = np.sum(np.argmax(res, axis=1) == np.argmax(test_label, axis=1)) / float(len(res))
            
            i,j = test_label.shape
            mse = np.sum((res-test_label)**2)/(i*j)
            print(f)
            print("Accuracy:")
            print(accuracy)
            print("Best MSE:")
            print(mse)
