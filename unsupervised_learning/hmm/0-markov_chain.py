#!/usr/bin/env python3
"""This module creates the markov_chain function"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain being in a particular state
    after a specified number of iterations

    Inputs:
    P - square 2D numpy.ndarray of shape (n, n) representing the transition
    matrix
        P[i, j] - probability of transitioning from state i to state j
        n - number of states in the markov chain
    s - numpy.ndarray of shape (1, n) representing the probability of
    starting in each state
    t - number of iterations that the markov chain has been through

    Returns:
    numpy.ndarray of shape (1, n) representing the probability of being in a
    specific state after t iterations, or None on failure
    """

    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    if not isinstance(s, np.ndarray) or len(s.shape) != 2:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[1]:
        return None

    if not isinstance(t, int) or t < 0:
        return None

    for _ in range(t):
        s = np.matmul(s, P)
    return s
