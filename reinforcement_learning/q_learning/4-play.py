#!/usr/bin/env python3
""" Module for creating the play function """

import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode.

    Inputs:
    - env: FrozenLakeEnv instance
    - Q: numpy.ndarray containing the Q-table
    - max_steps: Maximum number of steps in the episode

    Returns:
    - The total rewards for the episode
    """

    # Reset the environment and get the initial state
    state = env.reset()
    env.render()  # Render the initial state
    terminate = False

    # action_map = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}

    # Loop over the number of maximum steps
    for step in range(max_steps):
        # Choose the action with the highest Q-value for the current state
        action = np.argmax(Q[state, :])
        new_state, reward, terminate, _ = env.step(action)

        # Print the action taken (e.g., Down, Right)
        # print(f"  ({action_map[action]})")

        # Render the environment
        env.render()

        # Check if the episode has terminated
        if terminate:
            return reward

        # Update the current state to the new state
        state = new_state

    env.close()
    return reward
