#!/usr/bin/env python3
"""This module creates the pool_backward function"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network

    Inputs:
    * dA - numpy.ndarray of shape (m, h_new, w_new, c) containing the
        partial derivatives with respect to the output of the pooling layer
        * m - number of examples
        * h_new - height of the output
        * w_new - width of the output
        * c - number of channels
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
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros((m, h_prev, w_prev, c))
    for ex in range(m):
        for kernel_index in range(c):
            for h in range(h_new):
                for w in range(w_new):
                    i = h * sh
                    j = w * sw
                    if mode is 'max':
                        pool = A_prev[ex, i: i + kh, j: j + kw, kernel_index]
                        mask = np.where(pool == np.max(pool), 1, 0)
                    elif mode is 'avg':
                        mask = np.ones((kh, kw))
                        mask /= (kh * kw)
                    dA_prev[ex, i: i + kh, j: j + kw, kernel_index] += (
                        mask * dA[ex, h, w, kernel_index])
    return dA_prev
