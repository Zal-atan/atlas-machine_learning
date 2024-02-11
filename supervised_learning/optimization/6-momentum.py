#!/usr/bin/env python3
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network in tensorflow using
    the gradient descent with momentum optimization algorithm.

    Inputs:
    loss - loss of the network
    alpha - learning rate
    beta1 - momentum weight

    Returns:
    The momentum optimization operation
    """
    optimized = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
    return optimized
