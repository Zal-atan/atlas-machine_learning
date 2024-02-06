#!/usr/bin/env python3
"""
Create a function create_layer(prev, n, activation):
"""
import tensorflow.compat.v1 as tf
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for the neural network

    Inputs:
    prev - tensor output of the previous layer
    n - number of nodes in the layer to create
    activation - activation function that the layer should use

    Return:
    tensor output of the layer
    """
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    return tf.layers.dense(prev, n , activation=activation,
                           kernel_initializer=weights)
