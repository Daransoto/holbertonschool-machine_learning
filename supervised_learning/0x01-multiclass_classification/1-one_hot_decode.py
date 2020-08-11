#!/usr/bin/env python3
""" This module contains the function one_hot_decode. """
import numpy as np


def one_hot_decode(one_hot):
    """ one hot decode of one_hot """
    if type(one_hot) != np.ndarray or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
