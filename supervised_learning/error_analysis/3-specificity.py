#!/usr/bin/env python3
"""This module will create the specificity function"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    Inputs:
    confusion - confusion numpy.ndarray of shape (classes, classes) where
        row indices represent the correct labels and
        column indices represent the predicted labels
        classes - the number of classes

    Returns:
    numpy.ndarray of shape (classes,) containing the specificity of each class
    """

    all_instances = np.sum(confusion)
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    positives = np.sum(confusion, axis = 1)

    true_negatives = (all_instances - predicted_positives -
                      positives + true_positives)

    number_negatives = (all_instances - positives)

    true_negative_ratio = true_negatives / number_negatives

    return true_negative_ratio
