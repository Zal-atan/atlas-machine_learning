#!/usr/bin/env python3
""" This module creates the transition_layer function. """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected
    Convolutional Networks:

    Inputs:
    X - output from the previous layer
    nb_filters - integer representing the number of filters in X
    compression - compression factor for the transition layer

    Returns:
    The output of the transition layer and the number of filters
    within the output, respectively
    """

    output = K.layers.BatchNormalization()(X)
    output = K.layers.Activation('relu')(output)
    output = K.layers.Conv2D(int(nb_filters * compression), 1,
                             kernel_initializer='he_normal')(output)
    output = K.layers.AveragePooling2D(2)(output)

    return output, int(nb_filters * compression)
