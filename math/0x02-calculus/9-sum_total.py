#!/usr/bin/env python3
""" This module contains the function summation_i_squared. """


def summation_i_squared(n):
    """ Calculates the summation of i squared from 1 to n.
    n is the number to stop at. """
    if type(n) != int or n <= 0:
        return None
    return n * (n + 1) * (2*n + 1) // 6
