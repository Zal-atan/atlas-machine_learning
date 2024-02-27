#!/usr/bin/env python3
"""This module creates the predict function"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Tests a neural network:

    Inputs:
    network - network model to test
    data - input data to test the model with
    verbose - boolean that determines if output should be printed during the
        testing process

    Return:
    The prediction for the data
    """
    predict = network.predict(data, verbose=verbose)

    return predict
