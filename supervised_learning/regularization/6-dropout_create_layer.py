#!/usr/bin/env python3
"""This module will create the dropout_create_layer function"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout

    Inputs:
    prev - tensor containing the output of the previous layer
    n - number of nodes the new layer should contain
    activation - activation function that should be used on the layer
    keep_prob - probability that a node will be kept

    Returns:
    The output of the new layer
    """

    dropout = tf.layers.Dropout(rate=keep_prob)
    init = tf.variance_scaling_initializer(scale=2.0, mode="fan_avg")

    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=dropout)

    return layer(prev)
