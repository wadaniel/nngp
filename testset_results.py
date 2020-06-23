import keras
from keras.datasets import mnist, cifar10
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

num_classes = 10

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="cifar10")
parser.add_argument('--path', required=True)
opt = parser.parse_args()

predictions = np.load(opt.path)

if opt.dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif opt.dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_test_reg = keras.utils.to_categorical(y_test, num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)
train_label_mean = np.mean(y_train)

y_test_reg  -= train_label_mean

print(accuracy_score(y_test, predictions.argmax(axis=1)))
print(mean_squared_error(y_test_reg, predictions))