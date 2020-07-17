#!/usr/bin/env python3
"""
This module contains the function np_slice.
"""


def np_slice(matrix, axes={}):
    """
    Function that gets a slice of a matrix over specified axes.
    """
    shape = matrix.shape
    slices = []
    for i in range(len(shape)):
        args = axes.get(i, None)
        if args:
            slices.append(slice(*args))
        else:
            slices.append(slice(None))
    slices = tuple(slices)
    return matrix[slices]
