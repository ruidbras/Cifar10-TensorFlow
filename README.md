# CIFAR-10 Classification with Keras

## CIFAR-10

The CIFAR-10 dataset consists of 60 000 32x32 colour images in 10 classes, with 6 000 images per class.<br>
To get more info on this dataset check the tech report [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.<br>

## Frameworks

This repo contains the implementation of a deep learning model on CIFAR-10 dataset using [Keras](https://keras.io/) with [Tensorflow](https://www.tensorflow.org/) backend.

Keras is a high level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.

Tensorflow is an open source software library for high performance numerical computation. With its flexible architecture it becomes easy the deployment of computation across a variety of platforms such as CPUs, GPUs, and TPUs.

Make sure to check the most recent releases and compatibilities.

## Neural Networks

Additional images are created by rotating, shifting, zooming, etc. the original training examples. 


In my case, I reach over 90 percent classificaion accuracy on CIFAR_10, a dataset with 50 000 training images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck). I tried different operations, the best seemed to be rotations, horizontal flipping and vertical or horizontal shifts. The model is a VGG-type convolutional network with 6 conv layers and one dense fully connected layer before the output. Apart from image augmentation, batch normalisation (in each layer) and dropout in the dense layer is used.

The results without image augmentation: 82.8% test accuracy, clearly overfitting!


