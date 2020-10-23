#!/usr/bin/env python3
"""
This module contains the functions determinant, minor, cofactor, and adjugate.
"""


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


def minor(matrix):
    """
    Calculates the minor matrix of a matrix.
    matrix is a list of lists whose minor matrix will be calculated.
    If matrix is not a list of lists, raises a TypeError with the message
     matrix must be a list of lists.
    If matrix is not square or is empty, raises a ValueError with the message
     matrix must be a non-empty square matrix.
    Returns: the minor matrix of matrix.
    """
    if (type(matrix) is not list or matrix == [] or
       any([type(el) != list for el in matrix])):
        raise TypeError('matrix must be a list of lists')
    lm = len(matrix)
    if any([lm != len(row) for row in matrix]):
        raise ValueError('matrix must be a non-empty square matrix')
    if lm == 1:
        return [[1]]
    ans = []
    for i in range(lm):
        row = []
        for j in range(lm):
            f = [[n for k, n in enumerate(c) if k != j] for idx, c in
                 enumerate(matrix) if idx != i]
            row.append(determinant(f))
        ans.append(row)
    return ans


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix.
    matrix is a list of lists whose cofactor matrix will be calculated.
    If matrix is not a list of lists, raises a TypeError with the message
     matrix must be a list of lists.
    If matrix is not square or is empty, raises a ValueError with the message
     matrix must be a non-empty square matrix.
    Returns: the cofactor matrix of matrix.
    """
    return [[(-1) ** (i + j) * n for j, n in enumerate(row)]
            for i, row in enumerate(minor(matrix))]


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix.
    matrix is a list of lists whose adjugate matrix will be calculated.
    If matrix is not a list of lists, raises a TypeError with the message
     matrix must be a list of lists.
    If matrix is not square or is empty, raisea a ValueError with the message
     matrix must be a non-empty square matrix.
    Returns: the adjugate matrix of matrix.
    """
    return [list(row) for row in zip(*cofactor(matrix))]
