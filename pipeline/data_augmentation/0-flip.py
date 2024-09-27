#!/usr/bin/env python3
""" Module for creating flip_image() function"""

import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally

    Input:
    image: 3D tf.Tensor containing the image to flip

    Returns:
    the flipped image
    """
    return tf.image.flip_left_right(image)
