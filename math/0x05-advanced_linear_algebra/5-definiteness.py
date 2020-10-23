#!/usr/bin/env python3
""" This module contains the function definiteness. """
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.
    matrix is a numpy.ndarray of shape (n, n) whose definiteness will be
     calculated.
    If matrix is not a numpy.ndarray, raises a TypeError with the message
     matrix must be a numpy.ndarray.
    If matrix is not a valid matrix, returns None.
    Return: the string Positive definite, Positive semi-definite,
     Negative semi-definite, Negative definite, or Indefinite if the matrix is
     positive definite, positive semi-definite, negative semi-definite,
     negative definite of indefinite, respectively.
    If matrix does not fit any of the above categories, returns None.
    """
    if type(matrix) != np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')
    try:
        eig = np.linalg.eigvals(matrix)
        if all(eig > 0):
            return 'Positive definite'
        elif all(eig >= 0):
            return 'Positive semi-definite'
        elif all(eig < 0):
            return 'Negative definite'
        elif all(eig <= 0):
            return 'Negative semi-definite'
        else:
            return 'Indefinite'
    except Exception:
        return None
