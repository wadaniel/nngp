import argparse
import csv
import numpy as np
from load_dataset import load_cifar10
from sklearn.metrics import mean_squared_error

def computeMSOD(outNN, outNNGP):
    return mean_squared_error(outNN, outNNGP)

def computeMSE(outNN, test_label):
    i,j = test_label.shape
    mse = np.sum((outNN-test_label)**2)/(i*j)
    return mse

def readNNGPFile(fname):
    f = np.load("/tmp/nngp/" + fname)
    return f

def readNNFile(fname):
    f = np.load("./experiments/" + fname)
    return f

def writeMSOD(fname, nnname, msod):
    with open(fname, 'a') as file:
        fieldnames = ['nn_name', 'val']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({'nn_name': nnname, 'val': msod})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fout', required=True, type=str)
    parser.add_argument('--fnn', required=True, type=str)
    parser.add_argument('--fnngp', required=True, type=str)
    
    (_, _, _, _, _, test_label) = load_cifar10(10000, mean_subtraction=True)
    opt = parser.parse_args()
    
    outnngp = readNNGPFile(opt.fnngp)
    outnn   = readNNFile(opt.fnn)
    
    msod = computeMSOD(outnn, outnngp)
    print(test_label.shape)
    print(outnn.shape)
    mse      = computeMSE(outnn, test_label)
    msebase  = computeMSE(outnngp, test_label)

    print("MSOD "+opt.fnn)
    print(msod)
    print(mse)
    print(msebase)
