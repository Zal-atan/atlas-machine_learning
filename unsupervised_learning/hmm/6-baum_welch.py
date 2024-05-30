#!/usr/bin/env python3
"""This module creates the baum-welch algorithm"""
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
    # P = np.sum(F[:, -1])

    return F


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
    # P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * bPP[:, 0])

    return bPP


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model

    Inputs:
    Observations - numpy.ndarray of shape (T,) that contains the index of the
    observation
        T - number of observations
    Transition - numpy.ndarray of shape (M, M) that contains the initialized
    transition probabilities
        M - number of hidden states
    Emission - numpy.ndarray of shape (M, N) that contains the initialized
    emission probabilities
        N - number of output states
    Initial - numpy.ndarray of shape (M, 1) that contains the initialized
    starting probabilities
    iterations - number of times expectation-maximization should be
    performed

    Returns:
    Converged Transition, Emission, or None, None on failure
    """

    T = len(Observations)
    N = Emission.shape[0]

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    # Iteration for the expectation-maximization process
    for n in range(iterations):
        # Forward pass
        alpha = forward(Observations, Emission, Transition, Initial)
        # Backward pass
        beta = backward(Observations, Emission, Transition, Initial)

        # Initialize xi
        xi = np.zeros((N, N, T - 1))

        # Calculate xi for each time step
        for t in range(T - 1):
            denom1 = np.dot(alpha[:, t].T, Transition)
            denom2 = denom1 * Emission[:, Observations[t + 1].T]
            denom = np.dot(denom2, beta[:, t + 1])

            for i in range(N):
                numer1 = alpha[i, t] * Transition[i]
                numer2 = numer1 * Emission[:, Observations[t + 1].T]
                numer = numer2 * beta[:, t + 1].T
                xi[i, :, t] = numer / denom

        # Calculate gamma
        gamma = np.sum(xi, axis=1)

        # Update Transition matrix
        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Update gamma for the last time step
        gamma = np.hstack(
            (gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        denom = np.sum(gamma, axis=1)

        # Update Emission matrix
        for s in range(Emission.shape[1]):
            Emission[:, s] = np.sum(gamma[:, Observations == s], axis=1)

        Emission = np.divide(Emission, denom.reshape((-1, 1)))

    return Transition, Emission
