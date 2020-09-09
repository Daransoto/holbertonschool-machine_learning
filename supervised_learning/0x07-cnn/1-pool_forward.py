#!/usr/bin/env python3
""" This module contains the function pool_forward. """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
     the output of the previous layer.
        m is the number of examples.
        h_prev is the height of the previous layer.
        w_prev is the width of the previous layer.
        c_prev is the number of channels in the previous layer.
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel for
     the pooling.
        kh is the kernel height.
        kw is the kernel width.
    stride is a tuple of (sh, sw) containing the strides for the pooling.
        sh is the stride for the height.
        sw is the stride for the width.
    mode is a string containing either max or avg, indicating whether to
     perform maximum or average pooling, respectively.
    Returns: the output of the pooling layer.
    """
    kh, kw = kernel_shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    sh, sw = stride
    if mode == 'max':
        pool = np.max
    else:
        pool = np.average
    ansh = int((h_prev - kh) / sh + 1)
    answ = int((w_prev - kw) / sw + 1)
    ans = np.zeros((m, ansh, answ, c_prev))
    for i in range(ansh):
        for j in range(answ):
            x = i * sh
            y = j * sw
            ans[:, i, j, :] = pool(A_prev[:, x: x + kh, y: y + kw, :],
                                   axis=(1, 2))
    return ans
