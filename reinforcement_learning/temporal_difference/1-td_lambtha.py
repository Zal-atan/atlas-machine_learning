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

    # Create empty numpy array of zeros in shape of value estimate
    elig_trace = np.zeros_like(V)
    # env.seed(0)

    # Loop through episodes
    for ep in range(0, episodes):

        # Reset the environment for each new episode
        state = env.reset()

        # Loop through steps until done or max steps
        for step in range(0, max_steps):
            # Get action
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            # Temporal Difference Error
            # δ = r + γV(s') - V(s)

            delta = reward + (gamma * V[next_state]) - V[state]

            # Update eligibility trace and move to the next state
            elig_trace[state] += 1
            elig_trace = elig_trace * (gamma * lambtha)

            # Update Value Estimate
            V += delta * alpha * elig_trace
            # Getting worse results with next one
            # V[state] += delta * alpha * elig_trace[state]

            if done:
                break

            # Move state forward
            state = next_state

    return V
