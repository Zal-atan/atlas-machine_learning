#!/usr/bin/env python3
""" This module creates the BIC function"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the data set
    kmin - positive integer containing the minimum number of clusters to
    check for (inclusive)
    kmax - positive integer containing the maximum number of clusters to
    check for (inclusive)
    tol - non-negative float containing tolerance of the log likelihood, used
    to determine early stopping i.e. if the difference is less than or equal
    to tol you should stop the algorithm
    verbose - boolean that determines if you should print information about
    the algorithm

    Returns: best_k, best_result, l, b, or None, None, None, None
    best_k - best value for k based on its BIC
    best_result - tuple containing pi, m, S
    l - numpy.ndarray of shape (kmax - kmin + 1) containing the log likelihood
    for each cluster size tested
    b - numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value for
    each cluster size tested
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= 0 or X.shape[0] <= kmax:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    # Initialize lists for results
    best_k = []
    best_result = []
    log_likeli = []
    b = []

    n, d = X.shape

    for k in range(kmin, kmax + 1):
        pi, m, S, _, log_l = expectation_maximization(X, k, iterations, tol,
                                                      verbose)

        best_k.append(k)
        best_result.append((pi, m, S))
        log_likeli.append(log_l)

        # Calculate parameters in BIC
        cov_params = k * d * (d + 1) / 2.
        mean_params = k * d
        p = cov_params + mean_params + k - 1

        # Calculate BIC = p * ln(n) - 2 * l
        bic = p * np.log(n) - (2 * log_l)
        b.append(bic)

    b = np.array(b)
    log_likeli = np.array(log_likeli)

    # Determine best cluster
    best_value = np.argmin(b)

    best_k = best_k[best_value]
    best_result = best_result[best_value]

    return best_k, best_result, log_likeli, b
