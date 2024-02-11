#!/usr/bin/env python3
""" This module creates normalize(X, m, s): function"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (Standardizes) a matrix

    Inputs:
    X - numpy.ndarray of shape (m, nx) to normalize
        m - the number of data points
        nx - number of features
    m - numpy.ndarray of shape (nx,) that contains mean of all features of X
    s - numpy.ndarray of shape (nx,) that contains the standard deviation
        of all features of X

    Returns:
    The normalized X matrix
    """
    normalized = (X - m) / s
    return normalized
