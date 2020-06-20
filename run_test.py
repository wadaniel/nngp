#!/bin/python
import load_dataset

def load_data(dataset):

  if dataset == 'mnist':
    (train_image, train_label, valid_image, valid_label, test_image,
     test_label) = load_dataset.load_mnist(
         num_train=1000,
         mean_subtraction=True,
         random_roated_labels=False)

  elif dataset == 'cifar':
    (train_image, train_label, valid_image, valid_label, test_image,
     test_label) = load_dataset.load_cifar10(
         num_train=1000,
         mean_subtraction=True,
         random_roated_labels=False)
  else:
    raise NotImplementedError

if __name__ == '__main__':
#    load_data('mnist')
    load_data('cifar')


