#!/usr/bin/env python3
"""This module will make a definiteness() function for calculating the
definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.

    Input:
    matrix - list of lists whose determinant should be calculated, must be
        numpy array of shape (n, n)

    Returns:
    The definiteness of the matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.array_equal(matrix, matrix.T):
        return None

    w, v = np.linalg.eig(matrix)
    if np.all(w > 0):
        return "Positive definite"
    if np.all(w >= 0):
        return "Positive semi-definite"
    if np.all(w < 0):
        return "Negative definite"
    if np.all(w <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
