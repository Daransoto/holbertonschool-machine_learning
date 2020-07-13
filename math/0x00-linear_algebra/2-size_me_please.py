#!/usr/bin/env python3
"""
This module contains just the function matrix_shape.
"""


def matrix_shape(matrix):
    """
    Function to get the shape of a given matrix.
    """
    temp = matrix
    shape = []
    while type(temp) == list:
        shape.append(len(temp))
        temp = temp[0]
    if len(shape) == 1:
        shape.insert(0, 1)
    return shape
