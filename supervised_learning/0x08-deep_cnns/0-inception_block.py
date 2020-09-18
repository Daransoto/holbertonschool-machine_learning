#!/usr/bin/env python3
""" This module contains the function inception_block. """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in Going Deeper with Convolutions.
    A_prev is the output from the previous layer.
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, and FPP.
        F1 is the number of filters in the 1x1 convolution.
        F3R is the number of filters in the 1x1 convolution before the 3x3
         convolution.
        F3 is the number of filters in the 3x3 convolution.
        F5R is the number of filters in the 1x1 convolution before the 5x5
         convolution.
        F5 is the number of filters in the 5x5 convolution.
        FPP is the number of filters in the 1x1 convolution after the max
         pooling.
    All convolutions inside the inception block use a rectified linear
     activation (ReLU).
    Returns: the concatenated output of the inception block.
    """
    He = K.initializers.he_normal()
    F1, F3R, F3, F5R, F5, FPP = filters
    F1_layer = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), padding='same',
                               activation='relu',
                               kernel_initializer=He)(A_prev)
    F3R_layer = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1),
                                padding='same', activation='relu',
                                kernel_initializer=He)(A_prev)
    F3_layer = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                               activation='relu',
                               kernel_initializer=He)(F3R_layer)
    F5R_layer = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1),
                                padding='same', activation='relu',
                                kernel_initializer=He)(A_prev)
    F5_layer = K.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding='same',
                               activation='relu',
                               kernel_initializer=He)(F5R_layer)
    pool = K.layers.MaxPool2D(pool_size=(3, 3), padding='same',
                              strides=(1, 1))(A_prev)
    FPP_layer = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                                padding='same', activation='relu',
                                kernel_initializer=He)(pool)
    stack = K.layers.concatenate([F1_layer, F3_layer, F5_layer, FPP_layer])
    return stack
