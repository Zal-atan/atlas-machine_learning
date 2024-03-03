#!/usr/bin/env python3
"""This module creates the lenet5 function"""
import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using Keras

    Inputs:
    * X is a tf.placeholder of shape (m, 28, 28, 1)
        containing the input images for the network
        * m is the number of images

    Returns:
    K.Model compiled to use Adam optimization
    (with default hyperparameters) and accuracy metrics
    """
    init = K.initializers.he_normal()

    conv_layer1 = K.layers.Conv2D(filters=6, kernel_size=5,
                                  padding='same', activation='relu',
                                  kernel_initializer=init)(X)

    max_pool1 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv_layer1)

    conv_layer2 = K.layers.Conv2D(filters=16, kernel_size=5,
                                  padding='valid', activation='relu',
                                  kernel_initializer=init)(max_pool1)

    max_pool2 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv_layer2)

    flatten = K.layers.Flatten()(max_pool2)

    FC1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer=init)(flatten)

    FC2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer=init)(FC1)

    FC3 = K.layers.Dense(units=10, kernel_initializer=init,
                         activation='softmax')(FC2)

    model = K.models.Model(X, FC3)

    adam = K.optimizers.Adam()

    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
