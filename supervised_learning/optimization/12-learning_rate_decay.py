#!/usr/bin/env python3
""" This module creates learning_rate_decay(alpha, decay_rate, global_step,
decay_step) function
"""
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy

    Inputs:
    alpha - original learning rate
    decay_rate - weight used to determine the rate at which alpha will decay
    global_step - number of passes of gradient descent that have elapsed
    decay_step - number of passes of gradient descent that should
                 occur before alpha is decayed further

    Returns:
    The learning rate decay operation
    """
    learn_rate = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                             decay_rate, staircase=True)

    return learn_rate
