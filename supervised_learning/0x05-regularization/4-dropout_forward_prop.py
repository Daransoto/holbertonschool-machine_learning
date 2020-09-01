#!/usr/bin/env python3
""" This module contains the function dropout_forward_prop. """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.
    X is a numpy.ndarray of shape (nx, m) containing the input data for the
     network.
        nx is the number of input features.
        m is the number of data points.
    weights is a dictionary of the weights and biases of the neural network.
    L the number of layers in the network.
    keep_prob is the probability that a node will be kept.
    All layers except the last use the tanh activation function.
    The last layer uses the softmax activation function.
    Returns: a dictionary containing the outputs of each layer and the dropout
     mask used on each layer.
    """
    cache = {}
    cache['A0'] = X
    for i in range(L):
        layer = str(i + 1)
        prev = str(i)
        w = 'W' + layer
        b = 'b' + layer
        a = 'A' + layer
        ap = 'A' + prev
        drop = 'd' + layer
        z = np.matmul(weights[w], cache[ap]) + weights[b]
        if i == L - 1:
            cache[a] = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
        else:
            cache[a] = np.tanh(z)
            sh = cache[a].shape
            cache[drop] = (np.random.rand(sh[0], sh[1]) < keep_prob) * 1
            cache[a] = np.multiply(cache[a], cache[drop])
            cache[a] /= keep_prob
    return cache
