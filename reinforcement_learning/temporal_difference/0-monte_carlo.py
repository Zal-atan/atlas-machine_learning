#!/usr/bin/env python3
"""This module creates the monte_carlo function"""

import numpy as np
import gym


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, 
                gamma=0.99):
    """
    Performs the Monte Carlo Algorithm

    Inputs:
    env: openAI environment instance\\
    V: numpy.ndarray of shape (s,) containing the value estimate\\
    policy: function that takes in a state and returns the 
        next action to take\\
    episodes: total number of episodes to train over\\
    max_steps: maximum number of steps per episode\\
    alpha: learning rate\\
    gamma: discount rate

    Return:\\
    V: the updated value estimate
    """
    
    # Run through each episode
    for ep in range(0, episodes):
        rewards_sum = 0
        # Reset the environment for each new episode
        state = env.reset()
          # To store state-reward pairs
        episode_steps = []

        # Simulate an episode
        for step_num in range(0, max_steps):
            # Get action, step environment through action, store state and rew
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_steps.append([state, reward])

            # If the episode is done, exit loop
            if done:
                break

            state = next_state

        episode_steps = np.array(episode_steps, dtype=int)

        # Update value estimates using the Monte Carlo method (backward update)
        for t in reversed(range(0, len(episode_steps))):
            # Get the state and reward at time t
            state, reward = episode_steps[t]
            # update discounted rewards sum
            rewards_sum = gamma * rewards_sum + reward

            # Update value estimate if state not seen earlier in the episode
            if state not in episode_steps[:ep, 0]:
                # Update the value estimate using the learning rate
                V[state] = V[state] + alpha * (rewards_sum - V[state])

    return V
