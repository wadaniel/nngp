from __future__ import print_function

import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD, Adam
from keras import initializers
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import tensorflow as tf

import argparse
import numpy as np
from math import sqrt
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--depth', required=True, type=int)
parser.add_argument('--width', required=True, type=int)
parser.add_argument('--decay', default=0.01, type=float)
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--train_size', default=50000, type=int)
parser.add_argument('--dataset', default="cifar10")
parser.add_argument('--weight_var', required=True, type=float)
parser.add_argument('--bias_var', required=True, type=float)
parser.add_argument("--sub_mean", action='store_true', default=False)
opt = parser.parse_args()

batch_size = opt.batch_size
lr = opt.lr
depth = opt.depth
width = opt.width
num_classes = 10
epochs = 15000

path = "experiments/nn_{}_lr{}_batch{}_depth{}_width{}_size{}_w{}_b{}".format(opt.dataset, lr, batch_size, depth, width, opt.train_size, opt.weight_var, opt.bias_var)
print(path)
# the data, split between train and test sets
if opt.dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif opt.dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)[:opt.train_size]
x_test = x_test.reshape(x_test.shape[0], -1)
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
x_train /= 255
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# create scaler
scaler = MinMaxScaler(feature_range=(-1,1))
#scaler = StandardScaler()
# fit scaler on data
scaler.fit(x_train)
# apply transform
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
 
# convert class vectors to binary class matrices
y_train = y_train[:opt.train_size]
y_train = keras.utils.to_categorical(y_train, num_classes).astype('float64')
#y_train = -0.1 * (np.ones(y_train.shape) - y_train) + 0.9 * y_train
y_test_reg = keras.utils.to_categorical(y_test, num_classes).astype('float64')
#y_test_reg = -0.1 * (np.ones(y_test.shape) - y_test_reg) + 0.9 * y_test_reg

# subtract mean
if opt.sub_mean:
    train_image_mean = np.mean(x_train)
    train_label_mean = np.mean(y_train)
 
    x_train -= train_image_mean
    y_train -= train_label_mean
    x_test  -= train_image_mean
    y_test_reg  -= train_label_mean

print(sqrt(opt.weight_var/width))
print(type(sqrt(opt.weight_var/width)))
print(x_train.shape[1])
model = Sequential()
model.add(Dense(width, activation='relu', input_shape=(x_train.shape[1],), 
    kernel_regularizer=l2(opt.decay), bias_regularizer=l2(opt.decay),
    kernel_initializer=initializers.RandomNormal(stddev=sqrt(opt.weight_var/width)),
    bias_initializer=initializers.RandomNormal(stddev=sqrt(opt.bias_var))))

for i in range(depth - 1):
    model.add(Dense(width, activation='relu',
        kernel_regularizer=l2(opt.decay), bias_regularizer=l2(opt.decay),
        kernel_initializer=initializers.RandomNormal(stddev=sqrt(opt.weight_var/width)),
        bias_initializer=initializers.RandomNormal(stddev=sqrt(opt.bias_var))))

model.add(Dense(num_classes,
    kernel_regularizer=l2(opt.decay), bias_regularizer=l2(opt.decay),
    kernel_initializer=initializers.RandomNormal(stddev=sqrt(opt.weight_var/width)),
    bias_initializer=initializers.RandomNormal(stddev=sqrt(opt.bias_var))))

model.summary()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=100000,
    decay_rate=opt.decay)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)#_schedule)

model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error', 'categorical_accuracy'])

class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        accuracy = logs["categorical_accuracy"]
        if accuracy >= self.threshold:
            self.model.stop_training = True

callbacks = [
    EarlyStopping(monitor='loss', patience=50, verbose=0),
    MyThresholdCallback(threshold=1.0)
    #EarlyStopping(monitor='categorical_accuracy', baseline=1.0, patience=0)
]

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks)

score = model.evaluate(x_test, y_test_reg, verbose=0)
print('Test loss:', score[0])
predictions = model.predict(x_test)
print(accuracy_score(y_test, predictions.argmax(axis=1)))
print(predictions)
np.save(path + ".npy", predictions)
