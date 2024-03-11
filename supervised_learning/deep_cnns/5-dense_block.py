#!/usr/bin/env python3
""" This module creates the dense_block function. """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected
    Convolutional Networks:

    Inputs:
    X - output from the previous layer
    nb_filters - integer representing the number of filters in X
    growth_rate - growth rate for the dense block
    layers - number of layers in the dense block

    Returns:
    The concatenated output of each layer within the Dense Block and the number
    of filters within the concatenated outputs, respectively
    """

    for layer in range(layers):

        output = K.layers.BatchNormalization()(X)
        output = K.layers.Activation('relu')(output)
        output = K.layers.Conv2D(4 * growth_rate, 1,
                                 kernel_initializer='he_normal')(output)

        output = K.layers.BatchNormalization()(output)
        output = K.layers.Activation('relu')(output)
        output = K.layers.Conv2D(growth_rate, 3, padding='same',
                                 kernel_initializer='he_normal')(output)

        X = K.layers.concatenate([X, output])

        nb_filters += growth_rate

    return X, nb_filters
