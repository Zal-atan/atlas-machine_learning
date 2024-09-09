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
    for episode_num in range(0, episodes):
        cumulative_reward = 0
        state = env.reset()  # Reset the environment for each new episode
        episode_steps = []  # To store state-reward pairs

        # Simulate an episode
        for step_num in range(0, max_steps):
            action = policy(state)  # Get action from the policy
            next_state, reward, done, _ = env.step(action)  # Take action in environment
            episode_steps.append([state, reward])  # Store the state and reward

            # If the episode is done, exit loop
            if done:
                break

            state = next_state  # Update the state for the next step

        episode_steps = np.array(episode_steps, dtype=int)

        # Update value estimates using the Monte Carlo method (backward update)
        for t in reversed(range(0, len(episode_steps))):
            state, reward = episode_steps[t]  # Get the state and reward at time t
            cumulative_reward = gamma * cumulative_reward + reward  # Discounted sum of rewards

            # Update value estimate only if this state has not been seen earlier in the episode
            if state not in [episode_steps[i][0] for i in range(t)]:
                # Update the value estimate using the learning rate
                V[state] = V[state] + alpha * (cumulative_reward - V[state])

    return V

