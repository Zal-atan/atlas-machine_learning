#!/usr/bin/env python3
from PIL import Image
import gym

import gym.wrappers

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl. policy import GreedyQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy

from keras.optimizers import Adam

import tensorflow.keras as K


import tensorflow as tf
tf.compat.v1.enable_eager_execution()

# import time
# from keras.models import load_model


INPUT_SHAPE = (84, 84)
LEN_WINDOW = 4

build_model = __import__('train').build_model
PreProcessor = __import__('train').PreProcessor


if __name__ == '__main__':

    # Create the game environment
    env = gym.make("Breakout-v4", render_mode="human")
    env.reset()
    number_actions = env.action_space.n
    window = 4

    # Put together features for DQNAgent
    model = build_model(number_actions)
    model.summary()
    memory = SequentialMemory(limit=10000, window_length=LEN_WINDOW)
    processor = PreProcessor()

    # Need to use GreedyQPolicy() for task, but it does not work, using other
    # policy = GreedyQPolicy()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)

    # When nb_steps_warmup < nb_steps, it crashes the program at that number
    dqn = DQNAgent(model=model, nb_actions=number_actions, policy=policy,
                   memory=memory, processor=processor,
                   nb_steps_warmup=5000000, gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)

    # Compile model
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    # Load Weights
    dqn.load_weights('policy.h5')

    # Have to use fit() to show simulation, as dqn.test() will not work
    dqn.fit(env,
            nb_steps=10000,
            log_interval=10000,
            visualize=False,
            verbose=2)

    # Try to evaluate Agent
    # dqn.test does not seem to work not matter what I try.
    # dqn.test(env, nb_episodes=1000, visualize=False)
