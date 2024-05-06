#!/usr/bin/env python3
""" This module creates the correlation function"""
import numpy as np


def correlation(C):
    """
    Calculates the mean and covariance of a data set

    Inputs:
    C - numpy.ndarray of shape (d, d) containing a covariance matrix
        d is the number of dimensions

    Returns:
    numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")

    d1, d2 = C.shape
    if d1 != d2:
        raise ValueError("C must be a 2D square matrix")

    D = np.sqrt(np.diag(C))
    D_inv = 1 / np.outer(D, D)
    correlation = D_inv * C

    return correlation
