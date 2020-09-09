#!/usr/bin/env python3
""" This module contains the function pool_backward. """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.
    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
     partial derivatives with respect to the output of the pooling layer.
        m is the number of examples.
        h_new is the height of the output.
        w_new is the width of the output.
        c is the number of channels.
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
     output of the previous layer.
        h_prev is the height of the previous layer.
        w_prev is the width of the previous layer.
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel for
     the pooling.
        kh is the kernel height.
        kw is the kernel width.
    stride is a tuple of (sh, sw) containing the strides for the pooling.
        sh is the stride for the height.
        sw is the stride for the width.
    mode is a string containing either max or avg, indicating whether to
     perform maximum or average pooling, respectively.
    Returns: the partial derivatives with respect to the previous layer
     (dA_prev).
    """
    A_sh = A_prev.shape
    kh, kw = kernel_shape
    m, h_prev, w_prev, c_prev = A_sh
    sh, sw = stride
    _, h_new, w_new, c_new = dA.shape
    dA_prev = np.zeros(A_sh)
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                x = j * sh
                y = k * sw
                for l in range(c_new):
                    currP = A_prev[i, x: x + kh, y: y + kw, l]
                    currZ = dA[i, j, k, l]
                    if mode == "max":
                        general = np.zeros(kernel_shape)
                        maxV = np.amax(currP)
                        np.place(general, currP == maxV, 1)
                        dA_prev[i, x: x + kh, y: y + kw, l] += general * currZ
                    else:
                        avg = currZ / (kh * kw)
                        general = np.ones(kernel_shape)
                        dA_prev[i, x: x + kh, y: y + kw, l] += general * avg
    return dA_prev
