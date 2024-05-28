#!/usr/bin/env python3
"""This module creates the regularn function"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain

    Inputs:
    P - square 2D numpy.ndarray of shape (n, n) representing the transition
    matrix
        P[i, j] - probability of transitioning from state i to state j
        n - number of states in the markov chain

    Returns:
    numpy.ndarray of shape (1, n) containing the steady state probabilities,
    or None on failure
    """

    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not (P > 0).all():
        return None
    if not (P < 1).all():
        return None

    # Find Stead State Vector
    for i in range(100):
        SSV = np.linalg.matrix_power(P, i)

    return SSV[0, np.newaxis]
