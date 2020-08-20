#!/usr/bin/env python3
""" This module contains the function moving_average. """
import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set with bias correction.
    data is the list of data to calculate the moving average of.
    beta is the weight used for the moving average.
    Returns: a list containing the moving averages of data.
    """
    v = 0
    average = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        bias = 1 - beta ** (i + 1)
        average.append(v / bias)
    return average
