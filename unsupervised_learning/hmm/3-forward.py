#!/usr/bin/env python3
"""This module creates the forward function"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model

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

    Returns: P, F, or None, None on failure
    P - likelihood of the observations given the model
    F - numpy.ndarray of shape (N, T) containing the forward path probabilities
        F[i, j] - probability of being in hidden state i at time j given
        the previous observations
    """

    # Get T and N
    T = len(Observation)
    N = Emission.shape[0]

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    # Initialize F
    F = np.zeros((N, T))
    F[:, 0] = (Initial.T * Emission[:, Observation[0]])

    # Compute forward path probabilities
    for t in range(1, T):
        for n in range(N):
            Transitions = Transition[:, n]
            Emissions = Emission[n, Observation[t]]
            F[n, t] = np.sum(Transitions * F[:, t - 1] * Emissions)

    # Calculate likelihood
    P = np.sum(F[:, -1])

    return P, F
