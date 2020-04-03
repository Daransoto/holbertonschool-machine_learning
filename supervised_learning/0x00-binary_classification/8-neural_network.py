#!/usr/bin/env python3
"""
Creates a neural network.
"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """ Neural network class. """
    def __init__(self, nx, nodes):
        """ Initializer for the neural network. """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) != int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros(nx)
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
