#!/usr/bin/env python3
"""This module creates the conv_backward function"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural
    network

    Inputs:
    * dZ - numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the unactivated output of the
        convolutional layer
        * m - number of examples
        * h_new - height of the output
        * w_new - width of the output
        * c_new - number of channels in the output
    * A_prev - numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        * m - number of examples
        * h_prev - height of the previous layer
        * w_prev - width of the previous layer
        * c_prev - number of channels in the previous layer
    * W - numpy.ndarray of shape (kh, kw, c_prev, c_new)
        containing the kernels for the convolution
        * kh - filter height
        * kw - filter width
        * c_prev - number of channels in the previous layer
        * c_new - number of channels in the output
    * b - numpy.ndarray of shape (1, 1, 1, c_new) containing the
        biases applied to the convolution
    * padding - string that is either same or valid, indicating the type of
        padding used
    * stride - tuple of (sh, sw) containing the strides for the convolution
        * sh - stride for the height
        * sw - stride for the width

    Returns:
    Partial derivatives with respect to the previous layer (dA_prev),
    the kernels (dW), and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "valid":
        ph, pw = 0, 0
    elif padding == "same":
        ph = (((h_prev - 1) * sh) + kh - h_prev) // 2 + 1
        pw = (((w_prev - 1) * sw) + kw - w_prev) // 2 + 1

    padded_image = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    dA_prev = np.zeros((m, h_prev + (2 * ph), w_prev + (2 * pw), c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for j in range(c_new):
            for k in range(h_new):
                for x in range(w_new):
                    v_start = k * sh
                    v_end = v_start + kh
                    h_start = x * sw
                    h_end = h_start + kw
                    padded_slice = padded_image[i,
                                                v_start:v_end,
                                                h_start:h_end, :]
                    dA_prev[i, v_start:v_end, h_start:h_end,
                            :] += W[:, :, :, j] * dZ[i, k, x, j]
                    dW[:, :, :, j] += padded_slice * dZ[i, k, x, j]
    if padding == "same":
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    return dA_prev, dW, db
