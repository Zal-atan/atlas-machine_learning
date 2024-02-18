#!/usr/bin/env python3
"""This module will create the create_confusion_matrix function"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix:

    Inputs:
    labels -  one-hot numpy.ndarray of shape (m, classes) containing the
              correct labels for each data point
        m - number of data points
        classes - number of classes
    logits - one-hot numpy.ndarray (m, classes) containing predicted labels

    Returns:
    Confusion numpy.ndarray of shape (classes, classes) with
    row indices representing the correct labels and
    column indices representing the predicted labels
    """
    return np.matmul(labels.T, logits)
