#!/usr/bin/env python3
""" This module contains the function conv_forward. """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural
     network.
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
     the output of the previous layer.
        m is the number of examples.
        h_prev is the height of the previous layer.
        w_prev is the width of the previous layer.
        c_prev is the number of channels in the previous layer.
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
     kernels for the convolution.
        kh is the filter height.
        kw is the filter width.
        c_prev is the number of channels in the previous layer.
        c_new is the number of channels in the output.
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
     applied to the convolution.
    activation is an activation function applied to the convolution.
    padding is a string that is either same or valid, indicating the type of
     padding used.
    stride is a tuple of (sh, sw) containing the strides for the convolution.
        sh is the stride for the height.
        sw is the stride for the width.
    Returns: the output of the convolutional layer.
    """
    kh, kw, _, c_new = W.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2)
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2)
    else:
        ph = pw = 0
    padded = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)), mode='constant',
                    constant_values=0)
    ansh = int((h_prev + 2 * ph - kh) / sh + 1)
    answ = int((w_prev + 2 * pw - kw) / sw + 1)
    ans = np.zeros((m, ansh, answ, c_new))
    for i in range(ansh):
        for j in range(answ):
            for k in range(c_new):
                x = i * sh
                y = j * sw
                ans[:, i, j, k] = (padded[:, x: x + kh, y: y + kw, :] *
                                   W[:, :, :, k]).sum(axis=(1, 2, 3))
                ans[:, i, j, k] = activation(ans[:, i, j, k] + b[0, 0, 0, k])
    return ans
