#!/usr/bin/env python3
""" This module contains the function convolve_grayscale_same. """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.
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
    kh, kw = kernel.shape
    m, imh, imw = images.shape
    ph = (kh - 1) / 2
    pw = (kw - 1) / 2
    padded = np.pad(images, ((0,), (ph,), (pw,)))
    ans = np.zeros((m, imh, imw))
    for i in range(imh):
        for j in range(imw):
            ans[:, i, j] = (padded[:, i: i + kh, j: j + kw] *
                            kernel).sum(axis=(1, 2))
    return ans
