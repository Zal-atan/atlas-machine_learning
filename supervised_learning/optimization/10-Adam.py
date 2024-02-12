#!/usr/bin/env python3
""" This module creates create_Adam_op(loss, alpha, beta1, beta2, epsilon)
function
"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm:

    Inputs:
    loss - loss of the network
    alpha - learning rate
    beta1 - weight used for the first moment
    beta2 - weight used for the second moment
    epsilon - small number to avoid division by zero

    Returns:
    The Adam optimization operationy
    """
    optimized_Adam = tf.train.AdamOptimizer(alpha, beta1, beta2,
                                            epsilon).minimize(loss)

    return optimized_Adam
