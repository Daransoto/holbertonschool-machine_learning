#!/usr/bin/env python3
"""
This module contains the function matrix_transpose.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.
    Type matrix: list(list(int/float))
    Type return: list(list(int/float))
    """
    matrixT = []
    for i in range(len(matrix[0])):
        current = []
        for row in matrix:
            current.append(row[i])
        matrixT.append(current)
    return matrixT
