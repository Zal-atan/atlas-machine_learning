#!/usr/bin/env python3
"""
Create a function create_train_op(loss, alpha):
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network:

    Inputs:
    loss - the loss of the networks prediction
    alpha - the learning rater

    Returns:
    operation that trains the network using gradient descent
    """

    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
