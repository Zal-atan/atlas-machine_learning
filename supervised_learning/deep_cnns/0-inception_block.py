#!/usr/bin/env python3
""" This module creates the inception_block function. """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in Going Deeper with
    Convolutions (2014)

    Inputs:
    * A_prev - output from the previous layer
    * filters - tuple or list containing F1, F3R, F3,F5R, F5, FPP,
        respectively:
        ** F1 - number of filters in the 1x1 convolution
        ** F3R - number of filters in the 1x1 convolution before the 3x3
            convolution
        ** F3 - number of filters in the 3x3 convolution
        ** F5R - number of filters in the 1x1 convolution before the 5x5
            convolution
        ** F5 - number of filters in the 5x5 convolution
        ** FPP - number of filters in the 1x1 convolution after the max pooling

    Returns:
    The concatenated output of the inception block
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    conv_1x1 = K.layers.Conv2D(F1, 1, activation='relu')(A_prev)

    conv_3x3 = K.layers.Conv2D(F3R, 1, activation='relu')(A_prev)
    conv_3x3 = K.layers.Conv2D(F3, 3, padding='same',
                               activation='relu')(conv_3x3)

    conv_5x5 = K.layers.Conv2D(F5R, 1, activation='relu')(A_prev)
    conv_5x5 = K.layers.Conv2D(F5, 5, padding='same',
                               activation='relu')(conv_5x5)

    pooling = K.layers.MaxPool2D(3, 1, padding='same')(A_prev)
    pooling = K.layers.Conv2D(FPP, 1, activation='relu')(pooling)

    return K.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pooling])
