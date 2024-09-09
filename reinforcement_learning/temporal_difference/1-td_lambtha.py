#!/usr/bin/env python3
""" This module creates the td_lambtha function"""

import numpy as np
import gym


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm

    Inputs:\\
    env: openAI environment instance\\
    V: numpy.ndarray of shape (s,) containing the value estimate\\
    policy: function that takes in a state and returns the next
        action to take\\
    lambtha: eligibility trace factor\\
    episodes: total number of episodes to train over\\
    max_steps: maximum number of steps per episode\\
    alpha: learning rate\\
    gamma: discount rate

    Returns:\\
    V: the updated value estimate
    """
