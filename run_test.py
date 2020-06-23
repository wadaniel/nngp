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

  elif dataset == 'stl10':
    (train_image, train_label, valid_image, valid_label, test_image,
     test_label) = load_dataset.load_stl10(
         num_train=1000,
         mean_subtraction=True,
         random_roated_labels=False)
   
  else:
    raise NotImplementedError

  return train_image, train_label, valid_image, valid_label, test_image, test_label

if __name__ == '__main__':
    (train_image, train_label, valid_image, valid_label, test_image, test_label) = load_data('mnist')
    #(train_image, train_label, valid_image, valid_label, test_image, test_label) = load_data('cifar')

    print(train_image.shape)
    print(train_label.shape)
    print(train_label)
    print(valid_image.shape)
    print(valid_label.shape)
    print(test_image.shape)
    print(test_label.shape)




