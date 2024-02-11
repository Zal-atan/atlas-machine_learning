#!/usr/bin/env python3
""" This module creates shuffle_data(X, Y): function"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices in the same way

    Inputs:
    X - numpy.ndarray of shape (m, nx) to normalize
        m - the number of data points
        nx - number of features in X
    Y - numpy.ndarray of shape (m, ny) to normalize
        m - the same number of data points as in X
        ny - number of features in Y

    Returns:
    Shuffled X and Y matrices
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)

    return X[shuffle], Y[shuffle]
