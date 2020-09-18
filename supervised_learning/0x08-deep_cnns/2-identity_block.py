#!/usr/bin/env python3
""" This module contains the function identity_block. """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning for Image
     Recognition (2015).
    A_prev is the output from the previous layer.
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution.
        F3 is the number of filters in the 3x3 convolution.
        F12 is the number of filters in the second 1x1 convolution.
    All convolutions inside the block are followed by batch normalization along
     the channels axis and a rectified linear activation (ReLU), respectively.
    All weights use he normal initialization.
    Returns: the activated output of the identity block.
    """
    He = K.initializers.he_normal()
    F11, F3, F12 = filters
    F11_layer = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                                padding='same', kernel_initializer=He)(A_prev)
    F11_layer = K.layers.BatchNormalization(axis=3)(F11_layer)
    F11_layer = K.layers.Activation('relu')(F11_layer)
    F3_layer = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                               kernel_initializer=He)(F11_layer)
    F3_layer = K.layers.BatchNormalization(axis=3)(F3_layer)
    F3_layer = K.layers.Activation('relu')(F3_layer)
    F12_layer = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                                padding='same',
                                kernel_initializer=He)(F3_layer)
    F12_layer = K.layers.BatchNormalization(axis=3)(F12_layer)
    iden = K.layers.Add()([F12_layer, A_prev])
    iden = K.layers.Activation('relu')(iden)
    return iden
