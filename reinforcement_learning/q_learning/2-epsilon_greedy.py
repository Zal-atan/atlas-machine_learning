#!/usr/bin/env python3
""" Module for creating the epsilon_greedy function """

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.\\
    You should sample p with numpy.random.uniformn to determine
    if your algorithm should explore or exploit\\
    If exploring, you should pick the next action with numpy.random.randint
    from all possible actions\\

    Inputs:\\
    Q: numpy.ndarray containing the q-table\\
    state: current state\\
    epsilon: epsilon to use for the calculation

    Returns:
    the next action index
    """

    p = np.random.uniform(0, 1)

    # Exploration: Select a random action
    if p < epsilon:
        action = np.random.randint(Q.shape[1])

    # Exploitation: Select the action with highest Q-value for current state
    else:
        action = np.argmax(Q[state, :])

    return action
