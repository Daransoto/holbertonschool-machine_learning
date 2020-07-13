#!/usr/bin/env python3
def matrix_transpose(matrix):
    matrixT = []
    for i in range(len(matrix[0])):
        current = []
        for row in matrix:
            current.append(row[i])
        matrixT.append(current)
    return matrixT
