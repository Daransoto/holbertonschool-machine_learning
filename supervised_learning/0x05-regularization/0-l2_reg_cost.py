#!/usr/bin/env python3
""" This module contains the function l2_reg_cost. """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.
    cost is the cost of the network without L2 regularization.
    lambtha is the regularization parameter.
    weights is a dictionary of the weights and biases (numpy.ndarrays) of the
     neural network.
    L is the number of layers in the neural network.
    m is the number of data points used.
    Returns: the cost of the network accounting for L2 regularization.
    """
    W = 0
    for i in range(1, L + 1):
        W += np.linalg.norm(weights.get('W' + str(i)))
    return cost + W * lambtha / (2 * m)
