#!/usr/bin/env python3
""" Module for creating shear_image() function"""

import tensorflow as tf


def shear_image(image, intensity):
    """
    Randomly shears an image

    Input:
    image: 3D tf.Tensor containing the image to shear
    intensity: the intensity with which the image should be sheared

    Returns:
    the sheared image
    """
    return tf.keras.preprocessing.image.random_shear(
        image,
        intensity,
        row_axis=1,
        col_axis=2,
        channel_axis=0,
        fill_mode='nearest',
        cval=0.0,
        interpolation_order=1)
