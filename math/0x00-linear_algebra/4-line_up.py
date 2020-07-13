#!/usr/bin/env python3
"""
This module contains the function add_arrays.
"""


def add_arrays(arr1, arr2):
    """
    Function to add 2 arrays element-wise.
    """
    lenarr1 = len(arr1)
    if lenarr1 != len(arr2):
        return None
    ans = []
    for i in range(lenarr1):
        ans.append(arr1[i] + arr2[i])
    return ans
