#!/usr/bin/env python3
""" This module creates the maximization function"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the data set
    g - numpy.ndarray of shape (k, n) containing the posterior probabilities
    for each data point in each cluster

    Returns: pi, m, S, or None, None, None on failure
    pi - numpy.ndarray of shape (k,) containing the updated priors
    for each cluster
    m - numpy.ndarray of shape (k, d) containing the updated centroid means
    for each cluster
    S - numpy.ndarray of shape (k, d, d) containing the updated covariance
    matrices for each cluste
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    if False in np.isclose(g.sum(axis=0), np.ones((g.shape[1]))):
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    # Initialize arrays to hold pi, m and S
    pi = np.zeros(k)
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    # Update priors of eacg cluster
    pi = np.sum(g, axis=1) / n

    # Update centroid means of clusters
    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    # Update covariance matrices
    for i in range(k):
        S[i] = (g[i] * ((X - m[i]).T)) @ (X - m[i]) / np.sum(g[i])

    return pi, m, S
