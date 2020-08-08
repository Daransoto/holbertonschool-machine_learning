#!/usr/bin/env python3
""" Creates a deep neural network. """
import numpy as np


class DeepNeuralNetwork:
    """ Deep neural network class. """

    def __init__(self, nx, layers):
        """ Init method """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        else:
            if nx < 1:
                raise ValueError('nx must be a positive integer')

        if type(layers) != list:
            raise TypeError('layers must be a list of positive integers')
        else:
            if any(list(map(lambda x: x <= 0, layers))):
                raise TypeError('layers must be a list of positive integers')
            if len(layers) < 1:
                raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(len(layers)):
            key = "W{}".format(i + 1)
            if i == 0:
                self.weights[key] = np.random.randn(layers[i],
                                                    nx)*np.sqrt(2/nx)
                self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            else:
                self.weights[key] = (np.random.randn(layers[i],
                                                     layers[i - 1]) *
                                     np.sqrt(2/layers[i - 1]))
                self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Getter for L (Number of layers). """
        return self.__L

    @property
    def cache(self):
        """ Getter for cache. """
        return self.__cache

    @property
    def weights(self):
        """ Getter for weights. """
        return self.__weights

    def forward_prop(self, X):
        """ Forward propagation of the network. """
        self.__cache['A0'] = X
        for i in range(self.L):
            w = self.weights['W{}'.format(i + 1)]
            b = self.weights['b{}'.format(i + 1)]
            p_a = self.cache['A' + str(i)]
            A = 1 / (1 + np.exp(-(np.matmul(w, p_a) + b)))
            self.__cache["A" + str(i + 1)] = A
        return (self.__cache["A" + str(i)], self.cache)

    def cost(self, Y, A):
        """ Calculates the cost of the network. """
        m = Y.shape[1]
        c = np.sum(-Y * np.log(A) - (1 - Y) * np.log(1.0000001 - A)) / m
        return c

    def evaluate(self, X, Y):
        """ Evaluates the output of the network. """
        A, _ = self.forward_prop(X)
        c = self.cost(Y, A)
        return (np.where(A >= 0.5, 1, 0), c)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates the gradient descent. """
        m = Y.shape[1]
        oldW = self.weights.copy()
        for i in range(self.L, 0, -1):
            A = cache["A" + str(i)]
            if i == self.L:
                dz = A - Y
            else:
                dz = np.matmul(oldW["W" + str(i + 1)].T, dz) * A * (1 - A)
            dw = np.matmul(dz, cache["A" + str(i - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            w = self.weights["W" + str(i)]
            b = self.weights["b" + str(i)]
            self.__weights["W" + str(i)] = w - alpha * dw
            self.__weights["b" + str(i)] = b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the network. """
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
        return self.evaluate(X, Y)
