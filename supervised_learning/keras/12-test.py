#!/usr/bin/env python3
"""This module creates the test_mdoel function"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network:

    Inputs:
    network - network model to test
    data - input data to test the model with
    labels - correct one-hot labels of data
    verbose - boolean that determines if output should be printed during the
        testing process

    Return:
    Loss and accuracy of the model with the testing data, respectively
    """
    test = network.evaluate(data, labels, verbose=verbose)

    return test
