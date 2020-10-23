#!/usr/bin/env python3
""" This module contains the function determinant. """


def determinant(matrix):
    """
    Calculates the determinant of a matrix.
    matrix is a list of lists whose determinant will be calculated.
    If matrix is not a list of lists, raises a TypeError with the message
     matrix must be a list of lists.
    If matrix is not square, raises a ValueError with the message matrix must
     be a square matrix.
    The list [[]] represents a 0x0 matrix.
    Returns: the determinant of matrix.
    """
    if (type(matrix) is not list or matrix == [] or
       any([type(el) != list for el in matrix])):
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]]:
        return 1
    if len(matrix) != len(matrix[0]):
        raise ValueError('matrix must be a square matrix')
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for i, n in enumerate(matrix[0]):
        m = matrix[1:]
        filt = [[num for idx, num in enumerate(col) if idx != i] for col in m]
        det += (n * (-1) ** i * determinant(filt))
    return det
