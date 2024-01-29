#!/usr/bin/env python3
"""Creates a function one_hot_encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix

    Inputs:
    Y - numpy.ndarray containing numeric class labels
    classes - the maximum number of classes found in Y

    Returns:
    one-hot encoding of Y with shape (classes, m), or None
    """
    if not isinstance(classes, int) or not isinstance(Y, np.ndarray):
        return None

    try:
        one_hot = np.eye(classes)[Y].T
        return one_hot
    except Exception as e:
        return None
