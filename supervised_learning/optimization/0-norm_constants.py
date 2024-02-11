#!/usr/bin/env python3
""" Module for defining normalization_constanst(X) function:"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standarization) constants of a matrix

    Inputs:
    X - numpy.ndarray of shape (m, nx) to normalize
        m - the number of data points
        nx - number of features

    Returns:
    the mean and standard deviation of each feature, respectively
    """
    mean = np.mean(X, axis=0)
    stdDev = np.std(X, axis=0)
    return mean, stdDev
