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
    dot_product = np.dot(matrix, weight)

    # Numerical stability fix for softmax (subtract max to prevent overflow)
    dot_product -= np.max(dot_product)

    exp = np.exp(dot_product)
    return exp / np.sum(exp, axis=1, keepdims=True)


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

    # monte_carlo = policy(state, weight)
    # action = np.random.choice(len(monte_carlo[0]), p=monte_carlo[0])

    monte_carlo = policy(state, weight)

    # Sample an action based on the softmax policy
    action = np.random.choice(len(monte_carlo[0]), p=monte_carlo[0])

    # Compute the gradient of the softmax policy
    d_softmax = monte_carlo.copy()
    d_softmax[0, action] -= 1  # Subtract 1 for the selected action's gradient

    # Compute the gradient of the policy with respect to the weight matrix
    grad = np.dot(state.T, d_softmax)

    return action, grad
