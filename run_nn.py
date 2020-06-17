from __future__ import print_function

import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD, Adam
from keras import initializers
from keras.callbacks import EarlyStopping

import argparse
import numpy as np
from math import sqrt
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--depth', required=True, type=int)
parser.add_argument('--width', required=True, type=int)
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--train_size', default=50000, type=int)
parser.add_argument('--dataset', default="cifar10")
parser.add_argument('--weight_var', required=True, type=float)
parser.add_argument('--bias_var', required=True, type=float)
opt = parser.parse_args()

batch_size = opt.batch_size
lr = opt.lr
depth = opt.depth
width = opt.width
num_classes = 10
epochs = 10000

# the data, split between train and test sets
if opt.dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif opt.dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)[:opt.train_size]
x_test = x_test.reshape(x_test.shape[0], -1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = y_train[:opt.train_size]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_train = -0.1 * (np.ones(y_train.shape) - y_train) + 0.9 * y_train
y_test_reg = keras.utils.to_categorical(y_test, num_classes)
y_test_reg = -0.1 * (np.ones(y_test.shape) - y_test_reg) + 0.9 * y_test_reg

print(sqrt(opt.weight_var/width))
print(type(sqrt(opt.weight_var/width)))
print(x_train.shape[1])
model = Sequential()
model.add(Dense(width, activation='relu', input_shape=(x_train.shape[1],), 
    kernel_initializer=initializers.RandomNormal(stddev=sqrt(opt.weight_var/width)),
    bias_initializer=initializers.RandomNormal(stddev=sqrt(opt.bias_var))))

for i in range(depth - 1):
    model.add(Dense(width, activation='relu',
        kernel_initializer=initializers.RandomNormal(stddev=sqrt(opt.weight_var/width)),
        bias_initializer=initializers.RandomNormal(stddev=sqrt(opt.bias_var))))

model.add(Dense(num_classes,
    kernel_initializer=initializers.RandomNormal(stddev=sqrt(opt.weight_var/width)),
    bias_initializer=initializers.RandomNormal(stddev=sqrt(opt.bias_var))))

model.summary()

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error'])

callbacks = [
    EarlyStopping(monitor='loss', patience=10, verbose=0),
]

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,)

score = model.evaluate(x_test, y_test_reg, verbose=0)
print('Test loss:', score[0])
predictions = model.predict(x_test)
print(accuracy_score(y_test, predictions.argmax(axis=1)))
#np.save()