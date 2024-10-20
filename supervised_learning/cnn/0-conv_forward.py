#!/usr/bin/env python3
"""This module creates the conv_forward function"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural
    network

    Inputs:
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
    * activation - activation function applied to the convolution
    * padding - string that is either same or valid, indicating the type of
        padding used
    * stride - tuple of (sh, sw) containing the strides for the convolution
        * sh - stride for the height
        * sw - stride for the width

    Returns:
    Output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "valid":
        ph, pw = 0, 0
    elif padding == "same":
        ph = (((h_prev - 1) * sh) + kh - h_prev) // 2
        pw = (((w_prev - 1) * sw) + kw - w_prev) // 2

    images = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    height = ((h_prev + (2 * ph) - kh) // sh) + 1
    weight = ((w_prev + (2 * pw) - kw) // sw) + 1

    conv_matrix = np.zeros((m, height, weight, c_new))

    for i in range(height):
        for j in range(weight):
            for k in range(c_new):
                v_start = i * sh
                v_end = v_start + kh
                h_start = j * sw
                h_end = h_start + kw
                kernel = W[:, :, :, k]
                output = np.multiply(images[:, v_start:v_end, h_start:h_end],
                                     kernel)
                conv_matrix[:, i, j, k] = (np.sum(output, axis=(1, 2, 3)))
    return activation(conv_matrix + b)
