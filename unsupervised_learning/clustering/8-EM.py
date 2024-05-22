#!/usr/bin/env python3
""" This module creates the expectation_maximization function"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the data set
    k - positive integer containing the number of clusters
    iterations - positive integer containing the maximum number of iterations
    for the algorithm
    tol - non-negative float containing tolerance of the log likelihood, used
    to determine early stopping i.e. if the difference is less than or equal
    to tol you should stop the algorithm
    verbose - boolean that determines if you should print information about
    the algorithm

    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
    pi - numpy.ndarray of shape (k,) containing the updated priors
    for each cluster
    m - numpy.ndarray of shape (k, d) containing the updated centroid means
    for each cluster
    S - numpy.ndarray of shape (k, d, d) containing the updated covariance
    matrices for each cluster
    g - numpy.ndarray of shape (k, n) containing the probabilities for each
    data point in each cluster
    l - log likelihood of the model
    """
    # import pdb; pdb.set_trace()
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    loglikelihood = 0
    for i in range(iterations + 1):
        g, loglikelihood_end = expectation(X, pi, m, S)

        # Print every 10
        if verbose and i % 10 == 0:
            loglikelihood_round = round(loglikelihood_end, 5)
            print(
                "Log Likelihood after {} iterations: {}".format(
                    i, loglikelihood_round))

        # if loglikelihoods are within tol, end
        if abs(loglikelihood_end - loglikelihood) <= tol:
            if verbose:
                loglikelihood_round = round(loglikelihood_end, 5)
                print(
                    "Log Likelihood after {} iterations: {}".format(
                        i, loglikelihood_round))
            break

        if i < iterations:
            pi, m, S = maximization(X, g)

        loglikelihood = loglikelihood_end

    return pi, m, S, g, loglikelihood_end
