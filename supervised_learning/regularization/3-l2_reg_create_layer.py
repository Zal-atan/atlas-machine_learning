#!/usr/bin/env python3
"""This module will create the l2_reg_create_layer function"""
import tensorflow.compat.v1 as tf
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization

    Inputs:
    prev - tensor containing the output of the previous layer
    n - number of nodes the new layer should contain
    activation - activation function that should be used on the layer
    lambtha - L2 regularization parameter

    Returns:
    The output of the new layer
    """

    regularizer = tf.keras.regularizers.l2(lambtha)
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")

    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=init,
                                  kernel_regularizer=regularizer)
    return layer(prev)
