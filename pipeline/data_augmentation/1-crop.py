#!/usr/bin/env python3
""" Module for creating crop_image() function"""

import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image

    Input:
    image: 3D tf.Tensor containing the image to crop
    size: tuple containing the size of the crop

    Returns:
    the cropped image
    """
    return tf.image.random_crop(image, size)
