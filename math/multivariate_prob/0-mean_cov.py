#!/usr/bin/env python3
""" This module creates the mean_cov function"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the data set
        n is the number of data points
        d is the number of dimensions in each data point

    Returns:
    mean - numpy.ndarray of shape (1, d) containing the mean of the data set
    cov - numpy.ndarray of shape (d, d) containing the covariance matrix
        of the data set
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    cov = (1 / (n - 1)) * np.matmul(X.T - mean.T, X - mean)

    return mean, cov
