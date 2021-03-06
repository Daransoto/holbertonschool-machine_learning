#!/usr/bin/env python3
"""
This module contains the functions matrix_shape, deepcopy, and cat_matrices.
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
    return shape


def deepcopy(matrix):
    """
    Function that creates a deepcopy of a matrix.
    """
    stackM = [matrix]
    stackA = []
    while stackM:
        currentM = stackM.pop()
        if stackA:
            currentA = stackA.pop()
        else:
            currentA = []
            ans = currentA
        if type(currentM[0]) == list:
            for row in currentM:
                currentA.append([])
                stackM.append(row)
                stackA.append(currentA[-1])
        else:
            currentA.extend(currentM)
    return ans


def cat_matrices(mat1, mat2, axis=0):
    """
    Funtion to concatenate 2 matrices over a given axis.
    """
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    dim1 = len(shape1)
    dim2 = len(shape2)
    if not dim1 == dim2:
        return None
    for i in range(dim1):
        if i == axis:
            continue
        if shape1[i] != shape2[i]:
            return None
    cpMat1 = deepcopy(mat1)
    cpMat2 = deepcopy(mat2)
    auxM1 = [cpMat1]
    auxM2 = [cpMat2]
    while axis:
        helperM1 = []
        helperM2 = []
        for i in range(len(auxM1)):
            helperM1.extend(auxM1[i])
            helperM2.extend(auxM2[i])
        auxM1 = helperM1
        auxM2 = helperM2
        axis -= 1
    for i in range(len(auxM1)):
        auxM1[i].extend(auxM2[i])
    return cpMat1
