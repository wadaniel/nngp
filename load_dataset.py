# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loader for NNGP experiments.

Loading MNIST dataset with train/valid/test split as numpy array.

Usage:
mnist_data = load_dataset.load_mnist(num_train=50000, use_float64=True,
                                     mean_subtraction=True)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import copy
import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds

from keras.datasets import mnist, cifar10

from getstl10 import read_all_images, read_labels

def load_mnist(num_train=50000,
               use_float64=False,
               mean_subtraction=False,
               random_roated_labels=False):
  
  """Loads MNIST as numpy array."""
  # uncomment later
  flags = tf.app.flags
  FLAGS = flags.FLAGS

  flags.DEFINE_string('data_dir', '/tmp/nngp/data/',
                    'Directory for data.')


  from tensorflow.examples.tutorials.mnist import input_data
  data_dir = FLAGS.data_dir
  datasets = input_data.read_data_sets(
      data_dir, False, validation_size=10000, one_hot=True)
  mnist_data = _select_subset(
      datasets,
      num_train,
      use_float64=use_float64,
      mean_subtraction=mean_subtraction,
      random_roated_labels=random_roated_labels)

  return mnist_data


def load_cifar10(num_train=50000,
                 use_float64=False,
                 mean_subtraction=False,
                 random_roated_labels=False):
    """Loads CIFAR as numpy array."""
    (x_train, y_trainlabel), (x_test, y_testlabel) = cifar10.load_data()
  
    x_train = np.array([img.flatten() for img in x_train])
    x_test  = np.array([img.flatten() for img in x_test])
    
    y_train = np.zeros((len(x_train), 10))
    for idx, l in enumerate(y_trainlabel):
        y_train[idx, l] = 1
  
    y_test = np.zeros((len(x_test), 10))
    for idx, l in enumerate(y_testlabel):
        y_test[idx, l] = 1
    
    x_train = x_train.astype('float64')
    y_train = y_train.astype('float64')
    x_test  = x_test.astype('float64')
    y_test  = y_test.astype('float64')

    if (num_train+5000 > len(x_train)):
        print("Too many training points selected {0} + {1} > {2}".format(num_train, 5000, len(x_train)))
        sys.exit()
 
    x_train = x_train[:num_train]
    y_train = y_train[:num_train]

    x_valid = x_train[-5000:]
    y_valid = y_train[-5000:]

    train_image_mean = np.mean(x_train)
    train_label_mean = np.mean(y_train)
 
    if mean_subtraction:
        x_train -= train_image_mean
        y_train -= train_label_mean
        x_test  -= train_image_mean
        y_test  -= train_label_mean
        x_valid -= train_image_mean
        y_valid -= train_label_mean
     
    return (x_train, y_train,
            x_valid, y_valid,
            x_test, y_test)


def load_stl10(num_train=4500,
                 use_float64=False,
                 mean_subtraction=False,
                 random_roated_labels=False,
                 poor=False):

    DATA_PATH_TRAIN = '/tmp/nngp/data/stl10_binary/train_X.bin'
    LABEL_PATH_TRAIN = '/tmp/nngp/data/stl10_binary/train_y.bin'

    DATA_PATH_TEST = '/tmp/nngp/data/stl10_binary/test_X.bin'
    LABEL_PATH_TEST = '/tmp/nngp/data/stl10_binary/test_y.bin'

    x_train      = read_all_images(DATA_PATH_TRAIN)
    y_trainlabel = read_labels(LABEL_PATH_TRAIN)
    y_trainlabel = y_trainlabel - 1
    x_test      = read_all_images(DATA_PATH_TEST)
    y_testlabel = read_labels(LABEL_PATH_TEST)
    y_testlabel = y_testlabel - 1
  
    if poor:
      return (x_train, y_trainlabel, x_test, y_testlabel)

    x_train = np.array([img.flatten() for img in x_train])
    x_test  = np.array([img.flatten() for img in x_test])

    y_train = np.zeros((len(x_train), 10))
    for idx, l in enumerate(y_trainlabel):
        y_train[idx, l] = 1
  
    y_test = np.zeros((len(x_test), 10))
    for idx, l in enumerate(y_testlabel):
        y_test[idx, l] = 1
 
    x_train = x_train.astype('float64')
    y_train = y_train.astype('float64')
    x_test  = x_test.astype('float64')
    y_test  = y_test.astype('float64')

    if (num_train+500 > len(x_train)):
        print("Too many training points selected {0} + {1} > {2}".format(num_train, 500, len(x_train)))
        sys.exit()
 
    x_train = x_train[:num_train]
    y_train = y_train[:num_train]

    x_valid = x_train[-500:]
    y_valid = y_train[-500:]

    train_image_mean = np.mean(x_train)
    train_label_mean = np.mean(y_train)
 
    if mean_subtraction:
        x_train -= train_image_mean
        y_train -= train_label_mean
        x_test  -= train_image_mean
        y_test  -= train_label_mean
        x_valid -= train_image_mean
        y_valid -= train_label_mean
     
    return (x_train, y_train,
            x_valid, y_valid,
            x_test, y_test)

def _select_subset(datasets,
                         num_train=100,
                         classes=list(range(10)),
                         seed=9999,
                         sort_by_class=False,
                         use_float64=False,
                         mean_subtraction=False,
                         random_roated_labels=False):
  """Select subset of MNIST and apply preprocessing."""
  np.random.seed(seed)
  classes.sort()
  subset = copy.deepcopy(datasets)

  num_class = len(classes)
  num_per_class = num_train // num_class

  idx_list = np.array([], dtype='uint8')

  ys = np.argmax(subset.train.labels, axis=1)  # undo one-hot
  for c in classes:
    if datasets.train.num_examples == num_train:
      idx_list = np.concatenate((idx_list, np.where(ys == c)[0]))
    else:
      idx_list = np.concatenate((idx_list,
                                 np.where(ys == c)[0][:num_per_class]))
  if not sort_by_class:
    np.random.shuffle(idx_list)

  data_precision = np.float64 if use_float64 else np.float32

  train_image = subset.train.images[idx_list][:num_train].astype(data_precision)
  train_label = subset.train.labels[idx_list][:num_train].astype(data_precision)
  valid_image = subset.validation.images.astype(data_precision)
  valid_label = subset.validation.labels.astype(data_precision)
  test_image = subset.test.images.astype(data_precision)
  test_label = subset.test.labels.astype(data_precision)

  if sort_by_class:
    train_idx = np.argsort(np.argmax(train_label, axis=1))
    train_image = train_image[train_idx]
    train_label = train_label[train_idx]

  if mean_subtraction:
    train_image_mean = np.mean(train_image)
    train_label_mean = np.mean(train_label)
    train_image -= train_image_mean
    train_label -= train_label_mean
    valid_image -= train_image_mean
    valid_label -= train_label_mean
    test_image -= train_image_mean
    test_label -= train_label_mean

  if random_roated_labels:
    r, _ = np.linalg.qr(np.random.rand(10, 10))
    train_label = np.dot(train_label, r)
    valid_label = np.dot(valid_label, r)
    test_label = np.dot(test_label, r)

  return (train_image, train_label,
          valid_image, valid_label,
          test_image, test_label)

