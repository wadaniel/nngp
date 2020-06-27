import argparse
import csv
import numpy as np
from sklearn.metrics import mean_squared_error

def computeMSOD(outNN, outNNGP):
    return mean_squared_error(outNN, outNNGP)

def readFile(fname):
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
    opt = parser.parse_args()
    
    outnngp = readFile(opt.fnngp)
    outnn   = readFile(opt.fnn)
    
    msod = computeMSOD(outnn, outnngp)
    writeMSOD(opt.fout, opt.fnn, msod)

