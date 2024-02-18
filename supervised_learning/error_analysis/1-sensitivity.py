#!/usr/bin/env python3
"""This module will create the sensitivity function"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    Inputs:
    confusion - confusion numpy.ndarray of shape (classes, classes) where
        row indices represent the correct labels and
        column indices represent the predicted labels
        classes - the number of classes

    Returns:
    numpy.ndarray of shape (classes,) containing the sensitivity of each class
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=1)
