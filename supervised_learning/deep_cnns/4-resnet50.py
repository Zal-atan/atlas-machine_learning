#!/usr/bin/env python3
""" This module creates the resnet50 function. """
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in Deep Residual
    Learning for Image Recognition (2015):

    You can assume the input data will have shape (224, 224, 3)

    Returns:
    The keras model
    """

    input = K.Input(shape=(224, 224, 3))

    output = K.layers.Conv2D(64, 7, 2, padding='same',
                             kernel_initializer='he_normal')(input)
    output = K.layers.BatchNormalization()(output)
    output = K.layers.Activation('relu')(output)
    output = K.layers.MaxPooling2D(3, 2, padding='same')(output)

    # Convolutions 2.x -  3 blocks
    output = projection_block(output, [64, 64, 256], 1)
    output = identity_block(output, [64, 64, 256])
    output = identity_block(output, [64, 64, 256])

    # Convolutions 3.x - 4 blocks
    output = projection_block(output, [128, 128, 512])
    output = identity_block(output, [128, 128, 512])
    output = identity_block(output, [128, 128, 512])
    output = identity_block(output, [128, 128, 512])

    # Convolutions 4.x - 6 blocks
    output = projection_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])

    # Convolutions 5.x - 3 blocks
    output = projection_block(output, [512, 512, 2048])
    output = identity_block(output, [512, 512, 2048])
    output = identity_block(output, [512, 512, 2048])

    output = K.layers.AveragePooling2D(7)(output)
    output = K.layers.Dense(1000, activation='softmax')(output)

    model = K.models.Model(inputs=input, outputs=output)

    return model
