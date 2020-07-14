#!/usr/bin/env python3
"""
This module contains the function cat_matrices2D.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Function to concatenate 2 2D matrices over an axis.
    """
    rows = len(mat2)
    columns = len(mat2[0])
    if axis == 0:
        if columns != len(mat1[0]):
            return None
        ans = []
        for row in mat1:
            ans.append(row.copy())
        return ans + mat2
    elif axis == 1:
        if rows != len(mat1):
            return None
        ans = []
        for i in range(rows):
            ans.append(mat1[i] + mat2[i])
        return ans
    return None
