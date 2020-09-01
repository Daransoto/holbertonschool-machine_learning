#!/usr/bin/env python3
""" This module contains the function dropout_gradient_descent. """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization using
     gradient descent.
    Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
     correct labels for the data.
        classes is the number of classes.
        m is the number of data points.
    weights is a dictionary of the weights and biases of the neural network.
    cache is a dictionary of the outputs and dropout masks of each layer of the
     neural network.
    alpha is the learning rate.
    keep_prob is the probability that a node will be kept.
    L is the number of layers of the network.
    All layers use the tanh activation function except the last, which uses the
     softmax activation function.
    The weights of the network are updated in place.
    """
    m = Y.shape[1]
    for i in range(L - 1, -1, -1):
        layer = str(i + 1)
        prev = str(i)
        if i == L - 1:
            dz = cache['A' + str(L)] - Y
        else:
            dz = da * (1 - cache['A' + layer] ** 2)
            dz *= cache['D' + layer]
            dz /= keep_prob
        dw = np.matmul(dz, cache['A' + prev].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        da = np.matmul(weights['W' + layer].T, dz)
        weights['W' + layer] = weights['W' + layer] - alpha * dw
        weights['b' + layer] = weights['b' + layer] - alpha * db
