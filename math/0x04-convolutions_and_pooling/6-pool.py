#!/usr/bin/env python3
""" This module contains the function pool. """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.
    images is a numpy.ndarray with shape (m, h, w, c) containing multiple
        images.
        m is the number of images.
        h is the height in pixels of the images.
        w is the width in pixels of the images.
        c is the number of channels in the image.
    kernel_shape is a tuple of (kh, kw) containing the kernel shape for the
     pooling.
        kh is the height of the kernel.
        kw is the width of the kernel.
    stride is a tuple of (sh, sw).
        sh is the stride for the height of the image.
        sw is the stride for the width of the image.
    mode indicates the type of pooling.
        max indicates max pooling.
        avg indicates average pooling.
    Returns: a numpy.ndarray containing the pooled images.
    """
    kh, kw = kernel_shape
    m, imh, imw, c = images.shape
    sh, sw = stride
    if mode == 'max':
        pool = np.max
    else:
        pool = np.average
    ansh = int((imh - kh) / sh + 1)
    answ = int((imw - kw) / sw + 1)
    ans = np.zeros((m, ansh, answ, c))
    for i in range(ansh):
        for j in range(answ):
            x = i * sh
            y = j * sw
            ans[:, i, j, :] = pool(images[:, x: x + kh, y: y + kw, :],
                                   axis=(1, 2))
    return ans
