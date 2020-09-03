#!/usr/bin/env python3
""" This module contains the function convolve_grayscale_padding. """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.
    images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images.
        m is the number of images.
        h is the height in pixels of the images.
        w is the width in pixels of the images.
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the
     convolution.
        kh is the height of the kernel.
        kw is the width of the kernel.
    padding is a tuple of (ph, pw).
        ph is the padding for the height of the image.
        pw is the padding for the width of the image.
        the image is padded with 0s.
    Returns: a numpy.ndarray containing the convolved images.
    """
    kh, kw = kernel.shape
    m, imh, imw = images.shape
    ph, pw = padding
    padded = np.pad(images, ((0,), (ph,), (pw,)))
    ansh = imh + 2 * ph - kh + 1
    answ = imw + 2 * pw - kw + 1
    ans = np.zeros((m, ansh, answ))
    for i in range(ansh):
        for j in range(answ):
            ans[:, i, j] = (padded[:, i: i + kh, j: j + kw] *
                            kernel).sum(axis=(1, 2))
    return ans
