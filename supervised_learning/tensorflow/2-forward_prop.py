#!/usr/bin/env python3
"""
Create a function forward_prop(x, layer_sizes=[], activations=[]):
"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network

    Inputs:
    x - placeholder for the input data
    layer_sizes - list containing the number of nodes in each layer of network
    activations - list contaiing activation functions for each layer of network

    Returns:
    the prediction of the network in tensor form
    """
    for node in range(len(layer_sizes)):
        if node == 0:
            prediction = create_layer(x, layer_sizes[node], activations[node])
        else:
            prediction = create_layer(prediction, layer_sizes[node],
                                      activations[node])
    return prediction

