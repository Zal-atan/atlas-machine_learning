#!/usr/bin/env python3
"""Creates a function one_hot_decode"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels:

    Inputs:
    one_hot - one-hot encoded numpy.ndarray with shape (classes, m)
        classes is max number of classes
        m is the number of examples

    Returns:
    numpy.ndarray containing the numeric labels for each example, or None
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    array = one_hot.T.argmax(axis=1)
    return array
