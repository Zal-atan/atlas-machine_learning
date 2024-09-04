#!/usr/bin/env python3
from PIL import Image
import gym

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl. policy import GreedyQPolicy

import tensorflow.keras as K


import tensorflow as tf
tf.compat.v1.enable_eager_execution()

# import time
# from keras.models import load_model


INPUT_SHAPE = (84, 84)
LEN_WINDOW = 4

build_model = __import__('train').build_model
AtariProcessor = __import__('train').AtariProcessor


if __name__ == '__main__':

    # Create environment
    env = gym.make("Breakout-v4", render_mode='human')
    env.reset()
    num_actions = env.action_space.n

    # Define parts of DQNAgent
    model = build_model(num_actions)  # deep conv net
    memory = SequentialMemory(limit=1000000, window_length=LEN_WINDOW)
    processor = AtariProcessor()
    policy = GreedyQPolicy()
    # Put together DQN
    dqn = DQNAgent(model=model, nb_actions=num_actions,
                   processor=processor, memory=memory, policy=policy)
    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])

    # Load Weights
    dqn.load_weights('policy.h5')

    # Try to evaluate Agent
    # I have tried verbose = True, and not pu
    dqn.test(env, nb_episodes=100)
