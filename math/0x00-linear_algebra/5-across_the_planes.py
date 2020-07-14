#!/usr/bin/env python3
"""
This module contains the function add_matrices2D.
"""


def add_matrices2D(mat1, mat2):
    """
    Function to add 2 2D matrices element-wise.
    """
    rows = len(mat1)
    columns = len(mat1[0])
    if rows != len(mat2) or columns != len(mat2[0]):
        return None
    ans = []
    for i in range(rows):
        current = []
        for j in range(columns):
            current.append(mat1[i][j] + mat2[i][j])
        ans.append(current)
    return ans
