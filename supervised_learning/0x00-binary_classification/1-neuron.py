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
