#!/usr/bin/env python3
""" This module creates the train() function"""

import numpy as np
import matplotlib as plt
import gym
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000007, gamma=0.98):
    """
    Implements a full training

    Inputs:
    env: initial environment
    nb_episodes: number of episodes used for training
    alpha: the learning rate
    gamma: the discount factor

    Returns:
    all values of the score (sum of all rewards during one episode loop)
    """
    # Get the number of observations and actions from the environment
    observations = env.observation_space.shape[0]
    actions = env.action_space.n
    
    # Initialize random weights in correct shape
    weights = np.random.rand(observations, actions)

    # Track scores thru iterations
    scores_list = []

    for episode in range(0, nb_episodes + 1):
        # Reset environment and get initial state
        state = env.reset()[None, :]
        # Store gradients and rewards, initialize ep score
        gradients = []
        rewards = []
        episode_score = 0

        done = False

        while not done:
            # Get action and gradient based on current weights, then step
            action, gradient = policy_gradient(state, weights)
            next_state, reward, done, _ = env.step(action)

            gradients.append(gradient)
            rewards.append(reward)

            episode_score += reward

            state = next_state[None, :]

        # Convert rewards to np
        rewards = np.array(rewards)

        # Update weights based on gradients and discounted rewards
        for i in range(len(gradients)):
            # Had to lower alpha, it was growing too large to quickly
            learning = (alpha * gradients[i])
            # Had to slightly lower gamma, would get scores near inf, then 0
            discount = sum(gamma ** rewards[i:] * rewards[i:])
            # Update weights
            weights += learning * discount

        scores_list.append(episode_score)

        # Print episode scores, remove last before updating
        print("Episode: " + str(episode) + " Score: " + str(episode_score),
              end="\r", flush=False) 

    # Return scores_list
    return scores_list
    