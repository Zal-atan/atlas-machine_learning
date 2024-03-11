#!/usr/bin/env python3
""" This module creates the densenet121 function. """
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in Densely Connected
    Convolutional Networks:

    You can assume the input data will have shape (224, 224, 3)

    Inputs:
    growth_rate - growth rate for the dense block
    compression - compression factor for the transition layer

    Returns:
    The keras model
    """

    input = K.Input(shape=(224, 224, 3))

    # Convolution and Pooling
    output = K.layers.BatchNormalization()(input)
    output = K.layers.Activation('relu')(output)
    output = K.layers.Conv2D(64, 7, 2, padding='same',
                             kernel_initializer='he_normal')(output)
    output = K.layers.MaxPooling2D(3, 2, padding='same')(output)

    # Dense Block and Transition Layer 1
    output, nb_filters = dense_block(output, 64, growth_rate, 6)
    output, nb_filters = transition_layer(output, nb_filters, compression)

    # Dense Block and Transition Layer 2
    output, nb_filters = dense_block(output, nb_filters, growth_rate, 12)
    output, nb_filters = transition_layer(output, nb_filters, compression)

    # Dense Block and Transition Layer 3
    output, nb_filters = dense_block(output, nb_filters, growth_rate, 24)
    output, nb_filters = transition_layer(output, nb_filters, compression)

    # Dense Block 4
    output, nb_filters = dense_block(output, nb_filters, growth_rate, 16)

    # Classification Layer
    output = K.layers.AveragePooling2D(7)(output)

    output = K.layers.Dense(1000, activation='softmax')(output)

    model = K.models.Model(inputs=input, outputs=output)

    return model
