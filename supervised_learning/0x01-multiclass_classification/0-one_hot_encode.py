#!/usr/bin/env python3
""" This module contains the function one_hot_encode. """
import numpy as np


def one_hot_encode(Y, classes):
    """ one hot encoding of Y array with classes amount of classes. """
    ans = np.eye(classes)[Y]
    return ans.T
