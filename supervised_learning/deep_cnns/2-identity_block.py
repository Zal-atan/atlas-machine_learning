#!/usr/bin/env python3
""" This module creates the identity_block function. """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015):

    Inputs:
    * A_prev is the output from the previous layer
    * filters is a tuple or list containing F11, F3, F12, respectively:
        ** F11 is the number of filters in the first 1x1 convolution
        ** F3 is the number of filters in the 3x3 convolution
        ** F12 is the number of filters in the second 1x1 convolution

    Returns:
    The activated output of the identity block
    """

    F11, F3, F12 = filters

    # 1x1 convolution
    output = K.layers.Conv2D(F11, 1, 1, padding='same',
                             kernel_initializer='he_normal')(A_prev)
    output = K.layers.BatchNormalization()(output)
    output = K.layers.Activation('relu')(output)

    # 3x3 convolution
    output = K.layers.Conv2D(F3, 3, 1, padding='same',
                             kernel_initializer='he_normal')(output)
    output = K.layers.BatchNormalization()(output)
    output = K.layers.Activation('relu')(output)

    # 1x1 convolution
    output = K.layers.Conv2D(F12, 1, 1, padding='same',
                             kernel_initializer='he_normal')(output)
    output = K.layers.BatchNormalization()(output)

    # Add the input to the output
    output = K.layers.Add()([output, A_prev])

    # Return the activated output
    return K.activations.relu(output)
