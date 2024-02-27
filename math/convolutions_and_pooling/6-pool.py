#!/usr/bin/env python3
""" This module creates the convolve function"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs a convolution on images using multiple kernels

    Inputs:
    images - numpy.ndarray with shape (m, h, w, c) containing multiple images
        m - number of images
        h - height in pixels of the images
        w - width in pixels of the images
        c - number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing the kernel shape
        for the pooling
        * kh - height of the kernel
        * kw - width of the kernel
    stride is a tuple of (sh, sw)
        sh - stride for the height of the image
        sw - stride for the width of the image
    mode - indicates the type of pooling
        max - indicates max pooling
        avg -i ndicates average pooling

    Return:
    numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    height = (h - kh) // sh + 1
    width = (w - kw) // sw + 1

    conv_matrix = np.zeros((m, height, width, c))

    for x in range(height):
        for y in range(width):
            i = x * sh
            j = y * sw
            if mode == "avg":
                mode = "mean"
            operation = getattr(np, mode)
            conv_matrix[:, x, y, :] = operation(images[:,
                                                       i:i + kh,
                                                       j:j + kw,
                                                       :], axis=(1, 2))

    return conv_matrix
