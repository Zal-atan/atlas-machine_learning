#!/usr/bin/env python3
""" This module creates the expectation function"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the dataset
    pi - numpy.ndarray of shape (k,) containing the priors for each cluster
    m - numpy.ndarray of shape (d,) containing the centroid means for each
    cluster
    S - numpy.ndarray of shape (k, d, d) containing the covariance
    matrices for each cluster

    Returns: g, l, or None, None on failure
    g - numpy.ndarray of shape (k, n) containing the posterior probabilities
    for each data point in each cluster
    l - total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if d != m.shape[1] or d != S.shape[1] or d != S.shape[2]:
        return None, None
    if k != m.shape[0] or k != S.shape[0]:
        return None, None

    # Initialize arrays to hold Gaussian components and PDF
    gauss_comps = np.zeros((k, n))
    PDF = np.zeros((k, n))

    # Calculate the PDF values for each cluster and data point
    for i in range(k):
        try:
            PDF[i] = pi[i] * pdf(X, m[i], S[i])

        except BaseException:
            return None, None

    # Normalize the Gauss components(posterior probabilities)
    gauss_comps = PDF / np.sum(PDF, axis=0)

    # Calculate the total log likelihood
    log_likelihood = np.sum(np.log(np.sum(PDF, axis=0)))

    return gauss_comps, log_likelihood
