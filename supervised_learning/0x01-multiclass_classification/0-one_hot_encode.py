#!/usr/bin/env python3
""" This module contains the function one_hot_encode. """
import numpy as np


def one_hot_encode(Y, classes):
    """ one hot encoding of Y array with classes amount of classes. """
    if type(Y) != np.ndarray or type(classes) != int or classes <= 0\
       or classes < np.max(Y) or\
       any(map(lambda x: type(x) != int or x < 0, Y)):
        return None
    ans = np.eye(classes)[Y]
    return ans.T
