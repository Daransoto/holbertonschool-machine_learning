#!/usr/bin/env python3
""" This module contains the function l2_reg_gradient_descent. """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
     with L2 regularization.
    Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
     correct labels for the data.
        classes is the number of classes.
        m is the number of data points.
    weights is a dictionary of the weights and biases of the neural network.
    cache is a dictionary of the outputs of each layer of the neural network.
    alpha is the learning rate.
    lambtha is the L2 regularization parameter.
    L is the number of layers of the network.
    The neural network uses tanh activations on each layer except the last,
     which uses a softmax activation.
    """
    m = Y.shape[1]
    for i in range(L - 1, -1, -1):
        curr = str(i + 1)
        prev = str(i)
        if i == L - 1:
            dz = cache['A' + str(L)] - Y
        else:
            dz = da * (1 - cache['A' + curr] ** 2)
        Ap = cache['A' + prev]
        dw = (np.matmul(dz, Ap.T) + lambtha * weights['W' + curr]) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        da = np.matmul(weights['W' + curr].T, dz)
        weights['W' + curr] = weights['W' + curr] - alpha * dw
        weights['b' + curr] = weights['b' + curr] - alpha * db
