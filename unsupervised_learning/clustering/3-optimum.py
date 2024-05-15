#!/usr/bin/env python3
""" This module creates the optimum function"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the dataset that will be
    used for K-means clustering
        n - number of data points
        d - number of dimensions for each data point
    kmin - positive integer containing the minimum number of clusters to
    check for (inclusive)
    kmax - positive integer containing the maximum number of clusters to
    check for (inclusive)
    iterations - positive integer containing the maximum number of iterations
    for K-means

    Returns: results, d_vars, or None, None on failure
    results - list containing the outputs of K-means for each cluster size
    d_vars - list containing the difference in variance from the smallest
    cluster size for each cluster size
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None
        if not isinstance(iterations, int) or iterations < 1:
            return None, None
        if kmax is None:
            kmax = X.shape[0]
        if not isinstance(kmax, int) or kmax < 1:
            return None, None
        if not isinstance(kmin, int) or kmin < 1 or kmin >= X.shape[0]:
            return None, None
        if kmin >= kmax:
            return None, None

        # Create empty lists
        results = []
        vars = []
        dist_vars = []

        # Iterate over range of cluster sizes
        for k in range(kmin, kmax + 1):

            # Perform K-means clustering
            C, clss = kmeans(X, k, iterations)
            results.append((C, clss))

            # Calculate variance in current cluster size
            vars.append(variance(X, C))

        # Calculate the difference in variance from the smallest cluster size
        for var in vars:
            dist_vars.append(vars[0] - var)

        return results, dist_vars

    except BaseException:
        return None, None
