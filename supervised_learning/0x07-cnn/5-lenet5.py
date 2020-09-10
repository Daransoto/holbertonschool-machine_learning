#!/usr/bin/env python3
""" This module contains the function lenet5. """
import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras.
    X is a K.Input of shape (m, 28, 28, 1) containing the input images for the
     network.
        m is the number of images.
    The model consists of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding.
        Max pooling layer with kernels of shape 2x2 with 2x2 strides.
        Convolutional layer with 16 kernels of shape 5x5 with valid padding.
        Max pooling layer with kernels of shape 2x2 with 2x2 strides.
        Fully connected layer with 120 nodes.
        Fully connected layer with 84 nodes.
        Fully connected softmax output layer with 10 nodes.
    All layers requiring initialization initialize their kernels with the
     he_normal initialization method.
    All hidden layers requiring activation use the relu activation function.
    Returns: a K.Model compiled to use Adam optimization
     (with default hyperparameters) and accuracy metrics.
    """
    He = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                            activation='relu', kernel_initializer=He)(X)
    pool1 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                            activation='relu', kernel_initializer=He)(pool1)
    pool2 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    pool2 = K.layers.Flatten()(pool2)
    dense1 = K.layers.Dense(units=120, activation='relu',
                            kernel_initializer=He)(pool2)
    dense2 = K.layers.Dense(units=84, activation='relu',
                            kernel_initializer=He)(dense1)
    dense3 = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer=He)(dense2)
    network = K.Model(inputs=X, outputs=dense3)
    network.compile(optimizer=K.optimizers.Adam(),
                    loss='categorical_crossentropy', metrics=['accuracy'])
    return network
