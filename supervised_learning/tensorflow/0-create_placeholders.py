#!/usr/bin/env python3
"""
Create a function create_placeholders(nx, classes)
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Creates and returns two placeholders, x and y, for the neural network.

    Inputs:
    nx - number of feature columns in our data
    classes: number of classes in our classifier

    Returns:
    x - placeholder for the input data to the neural network
    y - is the placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
