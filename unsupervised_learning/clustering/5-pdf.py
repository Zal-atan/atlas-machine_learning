#!/usr/bin/env python3
""" This module creates the pdf function"""
import numpy as np


def pdf(X, m, S):
    """
    Tests for the optimum number of clusters by variance

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the data points whose
    PDF should be evaluated
    m - numpy.ndarray of shape (d,) containing the mean of the distribution
    S - numpy.ndarray of shape (d, d) containing the covariance of the
    distribution

    Returns: P, or None on failure
    P - numpy.ndarray of shape (n,) containing the PDF values for
    each data point
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    if S.shape[0] != S.shape[1]:
        return None

    if m.shape[0] != S.shape[0]:
        return None

    n, d = X.shape

    # Calculate the inverse and determinant of the covariance matrix
    inverse_S = np.linalg.inv(S)
    determinant_S = np.linalg.det(S)

    # Determinant must be positive
    if determinant_S <= 0:
        return None

    # Formula
    # p(x∣ μ,Σ) = (1 √(2π)d|Σ|)exp(−1/2(x−μ)T Σ−1(x−μ))

    exponent = -0.5 * np.sum(np.dot(X - m, inverse_S) * (X - m), axis=1)
    coefficient = 1 / np.sqrt((2 * np.pi) ** d * determinant_S)

    # Calculate PDF using coefficient and exponent
    pdf = coefficient * np.exp(exponent)

    # Avoid very small values by setting a min
    P = np.where(pdf < 1e-300, 1e-300, pdf)

    return P
