import argparse
import numpy as np
from sklearn.metrics import mean_squared_error

def computeMSOD(outNN, outNNGP):
    return mean_squared_error(outNN, outNNGP)

def readFile(fname):
    f = np.load(fname)
    #print(f)
    print(f.shape)
    return f

def writeMSOD(msod):
    print(msod)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fnn', required=True, type=str)
    parser.add_argument('--fnngp', required=True, type=str)
    opt = parser.parse_args()

    outnngp = readFile(opt.fnngp)
    outnn   = readFile(opt.fnn)

    msod = computeMSOD(outnn, outnngp)
    writeMSOD(msod)

