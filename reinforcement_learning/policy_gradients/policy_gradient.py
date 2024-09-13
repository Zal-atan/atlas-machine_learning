#!/usr/bin/env python3
""" Module creating the functions policy() and policy_gradient()"""

import numpy as np
import gym


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix

    Inputs:\\
    matrix: input matrix\\
    weight: weight of the input policy

    Returns:\\
    Policy
    """

    # Softmax using dot products and exponent of dot product
    dot_product = matrix.dot(weight)
    exponent = np.exp(dot_product)

    # Return the policy
    return exponent / np.sum(exponent)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and a
    weight matrix

    Inputs:\\
    state: matrix representing the current observation of the environment\\
    weight: matrix of random weight

    Returns:\\
    action, gradient (in this order)
    """

    monte_carlo = policy(state, weight)
    action = np.random.choice(len(monte_carlo[0]), p=monte_carlo[0])

    # Need to reshape the policy to build softmax, so we do that here
    reshape = monte_carlo.reshape(-1, 1)

    softmax = (np.diagflat(reshape) - np.dot(reshape, reshape.T))[action, :]

    log_derivative = softmax / monte_carlo[0, action]

    grad = state.T.dot(log_derivative[None, :])

    return action, grad
