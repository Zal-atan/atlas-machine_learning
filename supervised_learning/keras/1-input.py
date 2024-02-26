#!/usr/bin/env python3
"""This module creates the build_model function"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library

    Inputs:
    nx - number of input features to the network
    layers - list containing the number of nodes in each layer of the network
    activations - list containing the activation functions used for each layer
                  of the network
    lambtha - L2 regularization parameter
    keep_prob - probability that a node will be kept for dropout

    Return:
    The keras model
    """
    inputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(lambtha)

    X = K.layers.Dense(units=layers[0],
                       activation=activations[0],
                       kernel_regularizer=regularizer)(inputs)
    for layer in range(1, len(layers)):
        X = K.layers.Dropout(1 - keep_prob)(X)
        X = K.layers.Dense(units=layers[layer],
                           activation=activations[layer],
                           kernel_regularizer=regularizer)(X)
    model = K.Model(inputs=inputs, outputs=X)

    return model
