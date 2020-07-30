#!/usr/bin/env python3
""" This module contains the Exponential class. """


class Exponential:
    """ Class that represents an Exponential distribution. """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Constructor of the class. Sets the instance attribute lambtha as float.
        data is a list of the data to be used to estimate the distribution.
        lambtha is the expected number of occurences in a given time frame.
        """
        if data:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = 0.
            count = 0
            for element in data:
                if type(element) not in {int, float}:
                    raise TypeError('Each element in data must be a number')
                count += 1
                mean += element
            self.lambtha = count / mean
        else:
            if type(lambtha) not in {int, float} or lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period (x).
        """
        if x < 0:
            return 0
        return self.lambtha * Exponential.e ** (-self.lambtha * x)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period (x).
        """
        if x < 0:
            return 0
        return 1 - Exponential.e ** (-self.lambtha * x)
