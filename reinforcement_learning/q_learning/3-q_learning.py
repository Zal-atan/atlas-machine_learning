#!/usr/bin/env python3
""" Module for creating the train function """

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning\\
    When the agent falls in a hole, the reward should be updated to be -1

    Inputs:\\
    env: FrozenLakeEnv instance\\
    Q: numpy.ndarray containing the Q-table\\
    episodes: total number of episodes to train over\\
    max_steps: maximum number of steps per episode\\
    alpha: learning rate\\
    gamma: discount rate\\
    epsilon: initial threshold for epsilon greedy\\
    min_epsilon: minimum value that epsilon should decay to\\
    epsilon_decay: decay rate for updating epsilon between episodes\\

    Returns: Q, total_rewards\\
    Q: updated Q-table\\
    total_rewards: list containing the rewards per episode
    """

    # Stores rewards from each episode
    rewards = []

    for epi in range(episodes):
        # Reset the environment and retrieve the initial state
        state = env.reset()[0]
        terminate = False
        total_rewards = 0

        for step in range(max_steps):
            # Choose an action using the epsilon-greedy strategy
            action = epsilon_greedy(Q, state, epsilon)

            # result = env.step(action)
            # print(len(result))  # Check the number of returned values
            # print(result)       # See what is being returned

            # Perform the action in the environment and observe the outcome
            new_state, reward, terminate, _, _ = env.step(action)

            # print(state)
            # print(action)

            # Update the Q-value for the state-action pair
            Q[state, action] = Q[state, action] + alpha * \
                (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

            state = new_state

            # Check if the episode has terminated
            if terminate is True:
                # If the agent falls into a hole, set the total rewards to -1
                if reward == 0.0:
                    total_rewards = -1
                total_rewards += reward
                break

            total_rewards += reward

        # Decay epsilon after each episode to reduce exploration over time
        epsilon = min_epsilon + (1 - min_epsilon) * \
            np.exp(-epsilon_decay * epi)

        # Append the total rewards for this episode to the rewards list
        rewards.append(total_rewards)

    return Q, rewards
