#!/usr/bin/env python3
""" This module contains the function one_hot. """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.
    Returns: the one-hot matrix
    """
    return K.utils.to_categorical(labels, classes)
