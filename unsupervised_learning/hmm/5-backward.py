#!/usr/bin/env python3
"""This module creates the backward function"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model

    Inputs:
    Observation - numpy.ndarray of shape (T,) that contains the index of
    the observation
        T - number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
        Emission[i, j] - probability of observing j given the hidden state i
        N - number of hidden states
        M - number of all possible observations
    Transition - 2D numpy.ndarray of shape (N, N) containing the
    transition probabilities
        Transition[i, j] is the probability of transitioning from the hidden
        state i to j
    Initial - numpy.ndarray of shape (N, 1) containing the probability of
    starting in a particular hidden state

    Returns: Returns: P, B, or None, None on failure
    P - likelihood of the observations given the model
    B - numpy.ndarray of shape (N, T) containing the backward path
    probabilities
        B[i, j] - probability of generating the future observations from hidden
        state i at time j
    """

    # Get T and N
    T = len(Observation)
    N = Emission.shape[0]

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    # Initialize Backward Path Probabilities
    bPP = np.zeros((N, T))
    bPP[:, T - 1] = np.ones((N))

    # Compute BPP
    for t in range(T - 2, -1, -1):
        for n in range(N):
            Transitions = Transition[n, :]
            Emissions = Emission[:, Observation[t + 1]]
            bPP[n, t] = np.sum(Transitions * bPP[:, t + 1] * Emissions)

    # Calculate most likelihood
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * bPP[:, 0])

    return P, bPP
