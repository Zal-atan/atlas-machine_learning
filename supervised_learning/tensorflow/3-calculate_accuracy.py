#!/usr/bin/env python3
"""
Create a function calculate_accuracy(y, y_pred):
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of the prediction

    Inputs:
    y - placeholder for the loabels of the input data
    y_pred - tensor containing the networks predictions

    Returns:
    a tensor contaiing the decimal accuracy of the prediction
    """

    y = tf.math.argmax(y, axis=1)
    y_pred = tf.math.argmax(y_pred, axis=1)

    prediction = tf.equal(y, y_pred)
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return accuracy
