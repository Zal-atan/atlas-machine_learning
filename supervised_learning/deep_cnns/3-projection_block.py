#!/usr/bin/env python3
""" This module creates the projection_block function. """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015):

    Inputs:
    * A_prev - output from the previous layer
    * filters - tuple or list containing F11, F3, F12, respectively:
        ** F11 - number of filters in the first 1x1 convolution
        ** F3 - number of filters in the 3x3 convolution
        ** F12 - number of filters in the second 1x1 convolution
    * s - stride of the first convolution in both the main path and the

    Returns:
    The activated output of the projection block
    """

    F11, F3, F12 = filters

    # 1x1 convolution
    output = K.layers.Conv2D(F11, 1, s, padding='same',
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

    output2 = K.layers.Conv2D(F12, 1, s, padding='same',
                              kernel_initializer='he_normal')(A_prev)
    output2 = K.layers.BatchNormalization()(output2)

    # Add the input to the output
    output = K.layers.add([output, output2])

    # Return the activated output
    return K.activations.relu(output)
