#!/usr/bin/env python3
"""This module creates the viterbi function"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden markov
    model

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

    Returns: path, P, or None, None on failure
    path - list of length T containing the most likely sequence of
    hidden states
    P - probability of obtaining the path sequence
    """

    # Get T and N
    T = len(Observation)
    N = Emission.shape[0]

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    # Initialize path and Viterbi path and backpointer
    path = [0 for i in range(T)]
    V = np.zeros((N, T))
    V[:, 0] = (Initial.T * Emission[:, Observation[0]])
    bP = np.zeros((N, T))

    # Compute Viterbi Path
    for t in range(1, T):
        for n in range(N):
            Transitions = Transition[:, n]
            Emissions = Emission[n, Observation[t]]
            V[n, t] = np.amax(Transitions * V[:, t - 1] * Emissions)
            bP[n, t - 1] = np.argmax(Transitions * V[:, t - 1] * Emissions)

    # Calculate most likely last state
    maximum = np.argmax(V[:, T - 1])
    path[0] = maximum

    index = 1
    for i in range(T - 2, -1, -1):
        path[index] = int(bP[int(maximum), i])
        maximum = bP[int(maximum), i]
        index += 1

    path.reverse()
    P = np.amax(V[:, T - 1], axis=0)
    return path, P
