#!/usr/bin/env python3
""" This module defines the likelihood function"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given various hypothetical
    probabilities of developing severe side effects

    Inputs:
    x - number of patients that develop severe side effects
    n - total number of patients observed
    P - 1D numpy.ndarray containing the various hypothetical probabilities
        of developing severe side effects

    Return:
    1D numpy.ndarray containing the likelihood of obtaining the data, x and n,
        for each probability in P, respectively
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal\
                          to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    factorial = np.math.factorial
    numer = factorial(n)
    denom = factorial(x) * factorial(n - x)
    fact = numer / denom
    likelihood = fact * (np.power(P, x)) * (np.power((1 - P), (n - x)))

    return likelihood
