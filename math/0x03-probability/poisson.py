#!/usr/bin/env python3
""" This module contains the Poisson class. """


class Poisson:
    """ Class that represents a poisson distribution. """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Constructor of the class. Sets the instance attribute lambtha as float.
        data is a list of the data to be used to estimate the distribution.
        lambtha is the expected number of occurences in a given time frame.
        """
        if data is not None:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = 0.
            count = 0
            for element in data:
                if type(element) not in {int, float}:
                    raise TypeError("Each element in data must be a number")
                count += 1
                mean += element
            self.lambtha = mean / count
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes (k).
        """
        k = int(k)
        if k < 0:
            return 0
        return Poisson.e ** -self.lambtha * self.lambtha ** k / Poisson.fact(k)

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes (k).
        """
        k = int(k)
        if k < 0:
            return 0
        summation = 0
        for i in range(0, k + 1):
            summation += (self.lambtha ** i / Poisson.fact(i))
        return Poisson.e ** -self.lambtha * summation

    @staticmethod
    def fact(n):
        """ Calculates the factorial of a number. """
        if type(n) != int or n < 0:
            raise ValueError('n must be a positive integer or 0.')
        ans = 1
        for i in range(2, n + 1):
            ans *= i
        return ans
