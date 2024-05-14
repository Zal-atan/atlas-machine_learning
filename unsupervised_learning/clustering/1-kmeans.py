#!/usr/bin/env python3
""" This module creates the kmeans function"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the dataset that will be
    used for K-means clustering
        n - number of data points
        d - number of dimensions for each data point
    k - positive integer containing the number of clusters

    Returns:
    numpy.ndarray of shape (k, d) containing the initialized centroids for
    each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    min_val = X.min(axis=0)
    max_val = X.max(axis=0)

    centroid = np.random.uniform(min_val, max_val, size=(k, X.shape[1]))

    return centroid


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the dataset that will be
    used for K-means clustering
        n - number of data points
        d - number of dimensions for each data point
    k - positive integer containing the number of clusters
    iterations - positive integer containing the maximum number of
    iterations that should be performed

    Returns: C, clss, or None, None on failure
    C - numpy.ndarray of shape (k, d) containing centroid means for each
    cluster
    clss - numpy.ndarray of shape (n,) containing the index of the cluster
    in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initial random centroids
    centroids = initialize(X, k)

    # Calculate new cetroids
    for _ in range(iterations):

        # Assign Clusters based on closest centroid
        clss = np.argmin(
            (np.linalg.norm((X - centroids[:, None, :]), axis=2).T), axis=1)

        # Combine dataset with cluster assignments
        labels = np.concatenate(
            (X.copy(), np.reshape(
                clss, (n, 1))), axis=1)

        # Initialize an array to store the new centroids
        means = np.zeros((k, d))

        # Calculate the new centroids
        for i in range(k):
            temp = labels[labels[:, -1] == i]
            temp = temp[:, :d]

            if temp.size == 0:
                means[i] = initialize(X, 1)
            else:
                means[i] = np.mean(temp, axis=0)

        # Re-assign clusters with the updated centroids
        clss = np.argmin(np.linalg.norm(
            (X - means[:, None, :]), axis=2).T, axis=1)

        # Check for convergence (if centroids do not change)
        if np.array_equal(centroids, means):
            break

        centroids = means

    return centroids, clss
