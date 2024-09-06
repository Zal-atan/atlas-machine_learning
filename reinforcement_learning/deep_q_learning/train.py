#!/usr/bin/env python3

from PIL import Image
import numpy as np
import gym

from rl.agents import DQNAgent
from rl.policy import LinearAnnealedPolicy,EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam

import tensorflow as tf
tf.compat.v1.enable_eager_execution()


INPUT_SHAPE = (84, 84)
LENG_WINDOW = 4

class PreProcessor(Processor):
    """Takes the shape of the the Atari Game and configures it to a 
    Grayscale 84 x 84 pixel window. Makes the training much faster"""
    def process_observation(self, observation):
        """
        Converts the game's RGB observation into a grayscale image resized to 84x84 pixels.
        This reduces the input size, speeding up training.
        """
        assert observation.ndim == 3
        # resize image
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """
        Normalizes the batch of game frames by scaling pixel values to a range of [0, 1].
        This improves neural network performance during training.
        """
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """
        Clips the reward to the range [-1, 1] to stabilize training by reducing the effect of large rewards.
        """
        return np.clip(reward, -1., 1.)


def build_model(number_actions):
    """
    This builds the model, both for training and playing
    """
    input_shape = (LENG_WINDOW,) + INPUT_SHAPE
    train_model = Sequential()

    # Build Deep CNN 
    train_model.add(Permute((2, 3, 1), input_shape=input_shape))
    train_model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    train_model.add(Activation('relu'))
    train_model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    train_model.add(Activation('relu'))
    train_model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    train_model.add(Activation('relu'))
    train_model.add(Flatten())
    train_model.add(Dense(512))
    train_model.add(Activation('relu'))
    train_model.add(Dense(number_actions))
    train_model.add(Activation('linear'))

    return train_model


if __name__ == '__main__':

    # Create the game environment
    env = gym.make("Breakout-v4")
    env.reset()
    number_actions = env.action_space.n
    window = 4

    # Put together features for DQNAgent
    model = build_model(number_actions)
    model.summary()
    memory = SequentialMemory(limit=10000, window_length=LENG_WINDOW)
    processor = PreProcessor()

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

    # If want to keep training from previous training
    # dqn.load_weights('policy.h5')

    # Train model - if visualize=True, deprecation error
    dqn.fit(env,
            nb_steps=3500000,
            log_interval=10000,
            visualize=False,
            verbose=2)

    # Save weights
    dqn.save_weights('policy1.h5', overwrite=True)
