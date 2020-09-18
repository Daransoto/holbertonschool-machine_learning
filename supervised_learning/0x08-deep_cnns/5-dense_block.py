#!/usr/bin/env python3
""" This module contains the function dense_block. """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected Convolutional
     Networks.
    X is the output from the previous layer.
    nb_filters is an integer representing the number of filters in X.
    growth_rate is the growth rate for the dense block.
    layers is the number of layers in the dense block.
    Uses the bottleneck layers used for DenseNet-B.
    All weights use he normal initialization.
    All convolutions should be preceded by Batch Normalization and a rectified
     linear activation (ReLU), respectively.
    Returns: The concatenated output of each layer within the Dense Block and
     the number of filters within the concatenated outputs, respectively.
    """
    He = K.initializers.he_normal()
    for i in range(layers):
        layer = K.layers.BatchNormalization()(X)
        layer = K.layers.Activation('relu')(layer)
        layer = K.layers.Conv2D(filters=4*growth_rate, kernel_size=1,
                                padding='same', kernel_initializer=He)(layer)
        layer = K.layers.BatchNormalization()(layer)
        layer = K.layers.Activation('relu')(layer)
        layer = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                padding='same', kernel_initializer=He)(layer)
        X = K.layers.concatenate([X, layer])
        nb_filters += growth_rate
    return X, nb_filters
