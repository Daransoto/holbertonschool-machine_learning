#!/usr/bin/env python3
""" This module contains the Binomial class. """


class Binomial:
    """ Class that represents a Binomial distribution. """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Constructor of the class. Sets the instance attributes n and p.
        data is a list of the data to be used to estimate the distribution.
        n is the number of Bernoulli trials (int).
        p is the probability of a success (float).
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
            mean /= count
            variance = 0.
            for element in data:
                variance += ((mean - element) ** 2)
            variance /= count
            p = 1 - variance / mean
            self.n = round(mean / p)
            self.p = float(mean / self.n)
        else:
            if type(n) not in {int, float} or n <= 0:
                raise ValueError('n must be a positive value')
            self.n = round(n)
            if not 0 < p < 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.p = float(p)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes (k).
        """
        if type(k) not in {int, float}:
            raise TypeError('k must be a number')
        k = int(k)
        if k < 0:
            return 0
        n = self.n
        p = self.p
        comb = Binomial.combinations
        return comb(n, k) * p ** k * (1 - p) ** (n - k)

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes (k).
        """
        if type(k) not in {int, float}:
            raise TypeError('k must be a number')
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf

    @staticmethod
    def combinations(n, r):
        """
        Calculates the number of combinations nCr.
        """
        fact = Binomial.fact
        return fact(n) / (fact(r) * fact(n - r))

    @staticmethod
    def fact(n):
        """ Calculates the factorial of a number. """
        if type(n) != int or n < 0:
            raise ValueError('n must be a positive integer or 0.')
        ans = 1
        for i in range(2, n + 1):
            ans *= i
        return ans
