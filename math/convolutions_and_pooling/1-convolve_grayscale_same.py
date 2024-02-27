#!/usr/bin/env python3
""" This module creates the convolve_grayscale_same function"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    Inputs:
    images - numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
        * m - number of images
        * h - height in pixels of the images
        * w - width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
        for the convolution
        * kh - height of the kernel
        * kw - width of the kernel

    Return:
    numpy.ndarray containing the convolved images
    """
    m, height, width = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]

    pad_top_bottom = kh // 2
    pad_left_right = kw // 2

    conv_matrix = np.zeros((m, height, width))

    images = np.pad(images, ((0, 0), (pad_top_bottom, pad_top_bottom),
                             (pad_left_right, pad_left_right)))

    for x in range(height):
        for y in range(width):
            output = np.multiply(images[:, x:x + kh, y:y + kw], kernel)
            conv_matrix[:, x, y] = np.sum(output, axis=(1, 2))

    return conv_matrix
