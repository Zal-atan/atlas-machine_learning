#!/usr/bin/env python3
"""This module creates the pool_forward function"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network

    Inputs:
    * A_prev - numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        * m - number of examples
        * h_prev - height of the previous layer
        * w_prev - width of the previous layer
        * c_prev - number of channels in the previous layer
    * kernel_shape is a tuple of (kh, kw) containing the size
        of the kernel for the pooling
        * kh - filter height
        * kw - filter width
    * stride - tuple of (sh, sw) containing the strides for the convolution
        * sh - stride for the height
        * sw - stride for the width
    * mode is a string containing either max or avg, indicating whether to
        perform maximum or average pooling, respectively

    Returns:
    Output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    height = ((h_prev - kh) // sh) + 1
    weight = ((w_prev - kw) // sw) + 1

    conv_matrix = np.zeros((m, height, weight, c_prev))

    for i in range(height):
        for j in range(weight):
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            output = A_prev[:, v_start:v_end, h_start:h_end, :]
            if mode == "max":
                conv_matrix[:, i, j, :] = np.max(output, axis=(1, 2))
            elif mode == "avg":
                conv_matrix[:, i, j, :] = np.mean(output, axis=(1, 2))

    return conv_matrix
