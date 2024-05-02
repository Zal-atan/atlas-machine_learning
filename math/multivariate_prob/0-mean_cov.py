#!/usr/bin/env python3
""" This module creates the mean_cov function"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the data set
        n is the number of data points
        d is the number of dimensions in each data point

    Returns:
    mean - numpy.ndarray of shape (1, d) containing the mean of the data set
    cov - numpy.ndarray of shape (d, d) containing the covariance matrix
        of the data set
    """
