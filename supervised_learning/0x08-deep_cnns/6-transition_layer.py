#!/usr/bin/env python3
""" This module contains the function transition_layer. """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected Convolutional
     Networks.
    X is the output from the previous layer.
    nb_filters is an integer representing the number of filters in X.
    compression is the compression factor for the transition layer.
    Implements compression as used in DenseNet-C.
    All weights use he normal initialization.
    All convolutions are preceded by Batch Normalization and a rectified
     linear activation (ReLU), respectively.
    Returns: The output of the transition layer and the number of filters
     within the output, respectively.
    """
    He = K.initializers.he_normal()
    layer = K.layers.BatchNormalization()(X)
    layer = K.layers.Activation('relu')(layer)
    nb_filters = int(nb_filters * compression)
    layer = K.layers.Conv2D(filters=nb_filters, kernel_size=1, padding='same',
                            kernel_initializer=He)(layer)
    layer = K.layers.AveragePooling2D(pool_size=2, padding='same')(layer)
    return layer, nb_filters
