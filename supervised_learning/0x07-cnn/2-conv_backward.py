#!/usr/bin/env python3
""" This module contains the function conv_backward. """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
     partial derivatives with respect to the unactivated output of the
     convolutional layer.
        m is the number of examples.
        h_new is the height of the output.
        w_new is the width of the output.
        c_new is the number of channels in the output.
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
     the output of the previous layer.
        h_prev is the height of the previous layer.
        w_prev is the width of the previous layer.
        c_prev is the number of channels in the previous layer.
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
     kernels for the convolution.
        kh is the filter height.
        kw is the filter width.
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
     applied to the convolution.
    padding is a string that is either same or valid, indicating the type of
     padding used.
    stride is a tuple of (sh, sw) containing the strides for the convolution.
        sh is the stride for the height.
        sw is the stride for the width.
    Returns: the partial derivatives with respect to the previous layer
     (dA_prev), the kernels (dW), and the biases (db), respectively.
    """
    A_sh = A_prev.shape
    W_sh = W.shape
    m, h_prev, w_prev, c_prev = A_sh
    kh, kw, _, c_new = W_sh
    sh, sw = stride
    _, h_new, w_new, _ = dZ.shape
    if padding == 'same':
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2) + 1
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2) + 1
    else:
        ph = pw = 0
    padded = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)), mode='constant',
                    constant_values=0)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    dA = np.zeros(padded.shape)
    dW = np.zeros(W_sh)
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                x = j * sh
                y = k * sw
                for l in range(c_new):
                    currZ = dZ[i, j, k, l]
                    currP = padded[i, x: x + kh, y: y + kw, :]
                    dA[i, x: x + kh, y: y + kw, :] += currZ * W[:, :, :, l]
                    dW[:, :, :, l] += currP * currZ
    if padding == 'same':
        dA = dA[:, ph:h_prev, pw:w_prev, :]
    return dA, dW, db
