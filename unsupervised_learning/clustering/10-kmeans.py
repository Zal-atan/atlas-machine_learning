#!/usr/bin/env python3
""" This module creates the kmeans function"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the data set
    k - number of clusters

    Returns: C, clss
    C - numpy.ndarray of shape (k, d) containing the centroid means for
    each cluster
    clss - numpy.ndarray of shape (n,) containing the index of the cluster in
    C that each data point belongs to
    """
    C, clss, _ = sklearn.cluster.k_means(X, k)

    return C, clss
