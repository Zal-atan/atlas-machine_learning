#!/usr/bin/env python3
"""This module will create the f1_score function"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix

    Inputs:
    confusion - confusion numpy.ndarray of shape (classes, classes) where
        row indices represent the correct labels and
        column indices represent the predicted labels
        classes - the number of classes

    Returns:
    numpy.ndarray of shape (classes,) containing the F1 score of each class
    """

    prec = precision(confusion)
    sens = sensitivity(confusion)

    f1 = 2 * (prec * sens) / (prec + sens)
    return f1
