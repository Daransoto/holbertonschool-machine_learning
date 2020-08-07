#!/usr/bin/env python3
""" Creates a deep neural network. """
import numpy as np


class DeepNeuralNetwork:
    """ Deep neural network class. """
    def __init__(self, nx, layers):
        """ Initializer for the deep neural network. """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list:
            raise TypeError('layers must be a list of positive integers')
        self.__L = 0
        self.__cache = {}
        self.__weights = {}
        rand = np.random.randn
        for idx, neurons in enumerate(layers):
            if type(neurons) != int:
                raise TypeError('layers must be a list of positive integers')
            if idx == 0:
                self.__weights['w1'] = rand(neurons, nx) * np.sqrt(2 / nx)
            else:
                p = layers[idx - 1]
                r = rand(neurons, p)
                self.__weights["w{}".format(idx + 1)] = r * np.sqrt(2 / p)
            self.__L += 1
            self.__weights["b{}".format(idx + 1)] = np.zeros((neurons, 1))

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
        out = np.matmul(self.weights['w1'], X) + self.weights['b1']
        A = 1 / (1 + np.exp(-out))
        self.__cache['A1'] = A
        for i in range(1, self.L):
            w = self.weights['w{}'.format(i + 1)]
            b = self.weights['b{}'.format(i + 1)]
            out = np.matmul(w, A) + b
            A = 1 / (1 + np.exp(-out))
            self.__cache['A{}'.format(i + 1)] = A
        return (A, self.cache)

    def cost(self, Y, A):
        """ Calculates the cost of the network. """
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        c = np.sum(loss[0]) / Y.shape[1]
        return c

    def evaluate(self, X, Y):
        """ Evaluates the output of the network. """
        A, _ = self.forward_prop(X)
        c = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return (A, c)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates the gradient descent. """
        last = self.L
        wstr = 'w{}'.format(last)
        bstr = 'b{}'.format(last)
        a = self.cache['A{}'.format(last)]
        dz = a - Y
        p_a = self.cache['A{}'.format(last - 1)]
        currW = self.weights[wstr]
        self.__weights[wstr] -= (alpha * np.matmul(dz, p_a.T) / a.shape[1])
        self.__weights[bstr] -= (alpha * np.sum(dz) / a.shape[1])
        last -= 1

        while last > 0:
            wstr = 'w{}'.format(last)
            bstr = 'b{}'.format(last)
            a = self.cache['A{}'.format(last)]
            dz = np.matmul(currW.T, dz) * a * (1 - a)
            p_a = self.cache['A{}'.format(last - 1)]
            currW = self.weights[wstr]
            self.__weights[wstr] -= (alpha * np.matmul(dz, p_a.T) / a.shape[1])
            self.__weights[bstr] -= (alpha * np.sum(dz, axis=1, keepdims=True)
                                     / a.shape[1])
            last -= 1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the network. """
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        for i in range(iterations):
            self.evaluate(X, Y)
            self.gradient_descent(Y, self.cache, alpha)
        return self.evaluate(X, Y)
