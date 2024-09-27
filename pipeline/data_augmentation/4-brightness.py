#!/usr/bin/env python3
""" Module for creating change_brightness() function"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image

    Input:
    image: 3D tf.Tensor containing the image to rotate
    max_delta: the maximum amount the image should be brightened (or darkened)

    Returns:
    the altered image
    """
    return tf.image.random_brightness(image, max_delta)
