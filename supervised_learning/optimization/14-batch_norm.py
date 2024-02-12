#!/usr/bin/env python3
""" This module creates create_batch_norm_layer(prev, n, activation) function
"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow

    Inputs:
    prev - activated output of the previous layer
    n - number of nodes in the layer to be created
    activation - activation function that should be used on the output
                 of the layer

    Returns:
    Tensor of the activated output for the layer
    """

    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense_layer = tf.keras.layers.Dense(units=n, activation=None,
                                        kernel_initializer=init)

    z = dense_layer(prev)
    mean, variance = tf.nn.moments(z, 0)
    gamma = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.zeros(n), trainable=True)
    epsilon = 1e-8

    activate = activation(tf.nn.batch_normalization(z, mean, variance,
                                                    beta, gamma, epsilon))

    return activate
