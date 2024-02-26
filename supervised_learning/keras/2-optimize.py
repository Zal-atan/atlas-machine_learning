#!/usr/bin/env python3
"""This module creates the optimize_model function"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model with categorical crossentropy
    loss and accuracy metrics

    Inputs:
    network - model to optimize
    alpha - learning rate
    beta1 - first Adam optimization parameter
    beta2 - second Adam optimization parameter

    Return:
    None
    """
    adam_opt = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=adam_opt, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
