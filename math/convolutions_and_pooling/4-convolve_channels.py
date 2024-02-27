#!/usr/bin/env python3
""" This module creates the convolve_channels function"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels:

    Inputs:
    images - numpy.ndarray with shape (m, h, w, c) containing multiple
        grayscale images
        * m - number of images
        * h - height in pixels of the images
        * w - width in pixels of the images
        * c - number of channels in the image
    kernel - numpy.ndarray with shape (kh, kw) containing the kernel
        for the convolution
        * kh - height of the kernel
        * kw - width of the kernel
    padding - tuple of (ph, pw)
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            ph - padding for the height of the image
            pw - padding for the width of the image
        the image should be padded with 0’s
    stride - tuple of (sh, sw)
        sh - stride for the height of the image
        sw - stride for the width of the image

    Return:
    numpy.ndarray containing the convolved images
    """
    m, height, width = images.shape[0], images.shape[1], images.shape[2]
    c = images.shape[3]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]

    if padding == 'same':
        pad_top_bottom = (((height - 1) * sh) + kh - height) // 2 + 1
        pad_left_right = (((width - 1) * sh) + kw - width) // 2 + 1

    elif padding == 'valid':
        pad_top_bottom = 0
        pad_left_right = 0

    else:
        pad_top_bottom = padding[0]
        pad_left_right = padding[1]

    height = (height + (2 * pad_top_bottom) - kh) // sh + 1
    width = (width + (2 * pad_left_right) - kw) // sw + 1

    conv_matrix = np.zeros((m, height, width))

    images = np.pad(images, ((0, 0), (pad_top_bottom, pad_top_bottom),
                             (pad_left_right, pad_left_right), (0, 0)))

    for x in range(height):
        for y in range(width):
            i = x * sh
            j = y * sw
            output = np.multiply(images[:, i:i + kh, j:j + kw], kernel)
            conv_matrix[:, x, y] = np.sum(output, axis=(1, 2, 3))

    return conv_matrix
