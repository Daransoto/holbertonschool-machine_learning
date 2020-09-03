#!/usr/bin/env python3
""" This module contains the function convolve_grayscale_valid. """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.
    images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images.
        m is the number of images.
        h is the height in pixels of the images.
        w is the width in pixels of the images.
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the
     convolution.
        kh is the height of the kernel.
        kw is the width of the kernel.
    Returns: a numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ansh = h - kh + 1
    answ = w - kw + 1
    ans = np.zeros((m, ansh, answ))
    for i in range(ansh):
        for j in range(answ):
            ans[:, i, j] = (images[:, i: i + kh, j: j + kw] *
                            kernel).sum(axis=(1, 2))
    return ans
