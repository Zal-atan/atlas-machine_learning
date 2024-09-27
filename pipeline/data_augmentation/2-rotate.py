#!/usr/bin/env python3
""" Module for creating rotate_image() function"""

import tensorflow as tf


def rotate_image(image, size):
    """
    Rotates an image by 90 degrees counter-clockwise

    Input:
    image: 3D tf.Tensor containing the image to rotate

    Returns:
    the rotated image
    """
    return tf.image.rot90(image, size)
