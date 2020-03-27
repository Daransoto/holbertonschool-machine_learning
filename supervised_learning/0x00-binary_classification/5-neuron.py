#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification.
"""
import numpy as np


class Neuron:
    """ Neuron class. """
    def __init__(self, nx):
        """ Initializer for the neuron. """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter for W. """
        return self.__W

    @property
    def b(self):
        """ Getter for b. """
        return self.__b

    @property
    def A(self):
        """ Getter for A. """
        return self.__A

    def forward_prop(self, X):
        """ Forward propagation algorithm using sigmoid function. """
        out = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-out))
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the neuron. """
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        c = np.sum(loss[0]) / Y.shape[1]
        return c

    def evaluate(self, X, Y):
        """ Evaluates the output of the neuron. """
        A = self.forward_prop(X)
        c = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return (A, c)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates the gradient descent. """
        self.__W = self.__W - alpha * np.matmul(A - Y, X.T) / A.shape[1]
        self.__b = self.__b - np.sum(alpha * (A - Y)) / A.shape[1]
