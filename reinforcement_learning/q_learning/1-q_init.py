#!/usr/bin/env python3
""" Module for creating the q_init function """

import numpy as np


def q_init(env):
    """
    Initializes the Q-table

    Inputs:\\
    env: the FrozenLakeEnv instance

    Returns:
    the Q-table as a numpy.ndarray of zeros
    """

    action = env.action_space.n
    obv_space = env.observation_space.n

    return np.zeros((obv_space, action))
