#!/usr/bin/env python3
""" Module for creating change_hue() function"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Randomly changes the hue of an image

    Input:
    image: 3D tf.Tensor containing the image to rotate
    delta: the amount the hue should change

    Returns:
    the altered image
    """
    return tf.image.random_hue(image, delta)
