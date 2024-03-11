#!/usr/bin/env python3
""" This module creates the inception_network function. """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network as described in Going Deeper with
    Convolutions (2014):

    You can assume the input data will have shape (224, 224, 3)

    Returns:
    The keras model
    """

    input = K.Input(shape=(224, 224, 3))

    output = K.layers.Conv2D(64, 7, 2, padding='same',
                             activation='relu')(input)
    output = K.layers.MaxPool2D(3, 2, padding='same')(output)

    output = K.layers.Conv2D(64, 1, 1, padding='same',
                             activation='relu')(output)
    output = K.layers.Conv2D(192, 3, 1, padding='same',
                             activation='relu')(output)
    output = K.layers.MaxPool2D(3, 2, padding='same')(output)

    # Inception layers
    IL3a = inception_block(output, [64, 96, 128, 16, 32, 32])
    IL3b = inception_block(IL3a, [128, 128, 192, 32, 96, 64])
    output = K.layers.MaxPool2D(3, 2, padding='same')(IL3b)

    IL4a = inception_block(output, [192, 96, 208, 16, 48, 64])
    IL4b = inception_block(IL4a, [160, 112, 224, 24, 64, 64])
    IL4c = inception_block(IL4b, [128, 128, 256, 24, 64, 64])
    IL4d = inception_block(IL4c, [112, 144, 288, 32, 64, 64])
    IL4e = inception_block(IL4d, [256, 160, 320, 32, 128, 128])
    output = K.layers.MaxPool2D(3, 2, padding='same')(IL4e)

    IL5a = inception_block(output, [256, 160, 320, 32, 128, 128])
    IL5b = inception_block(IL5a, [384, 192, 384, 48, 128, 128])
    output = K.layers.AvgPool2D(7, 1)(IL5b)

    output = K.layers.Dropout(0.4)(output)
    output = K.layers.Dense(1000, activation='softmax')(output)

    return K.Model(input, output)
