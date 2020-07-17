#!/usr/bin/env python3
"""
This module contains the functions add_matrices and matrix_shape.
"""


def matrix_shape(matrix):
    """
    Function to get the shape of a matrix.
    """
    temp = matrix
    shape = []
    while type(temp) == list:
        shape.append(len(temp))
        temp = temp[0]
    if len(shape) == 1:
        shape.insert(0, 1)
    return shape


def add_matrices(mat1, mat2):
    """
    Function to add 2 matrices.
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    if type(mat1[0]) != list:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    stackM1 = [row for row in mat1]
    stackM2 = [row for row in mat2]
    stackSum = [[] for _ in mat1]
    ans = [row for row in stackSum]
    while stackM1:
        currentM1 = stackM1.pop()
        currentM2 = stackM2.pop()
        currentSum = stackSum.pop()
        if type(currentM1[0]) == list:
            stackM1.extend([row for row in currentM1])
            stackM2.extend([row for row in currentM2])
            currentSum.extend([[] for _ in currentM1])
            stackSum.extend(row for row in currentSum)
        else:
            for i in range(len(currentM1)):
                currentSum.append(currentM1[i] + currentM2[i])
    return ans
