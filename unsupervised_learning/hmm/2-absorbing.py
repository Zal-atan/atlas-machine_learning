#!/usr/bin/env python3
"""This module creates the absorbing function"""
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing

    Inputs:
    P - square 2D numpy.ndarray of shape (n, n) representing the transition
    matrix
        P[i, j] - probability of transitioning from state i to state j
        n - number of states in the markov chain

    Returns:
    True if it is absorbing, or False on failure
    """

    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    # if not (P > 0).all():
    #     return None
    # if not (P < 1).all():
    #     return False

    Diag = np.diagonal(P)
    if (Diag == 1).all():
        return True
    if not (Diag == 1).any():
        return False

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if (i == j) and (i + 1 < len(P)):
                if P[i + 1][j] == 0 and P[i][j + 1] == 0:
                    return False

    return True
