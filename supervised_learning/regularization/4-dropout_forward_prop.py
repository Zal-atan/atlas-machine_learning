#!/usr/bin/env python3
"""This module will create the dropout_forward_prop function"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization

    Inputs:
    X - numpy.ndarray of shape (nx, m) containing the input data for the network
        classes - number of classes
        m - number of data points
    weights - dictionary of the weights and biases of the neural network
    L - number of layers of the network
    keep_prob - probability that a node will be kept

    Return:
    Dictionary containing the outputs of each layer and the dropout
    mask used on each layer
    """

    cache = {'A0': X}

    for i in range(1, L + 1):
        W = weights['W{}'.format(i)]
        b = weights['b{}'.format(i)]
        Z = np.matmul(W, cache['A{}'.format(i - 1)]) + b

        if i == L:
            e = np.exp(Z)
            cache['A{}'.format(i)] = e / np.sum(e, axis=0, keepdims=True)

        else:
            dropout = np.random.binomial(1, keep_prob, size=Z.shape)
            cache['A{}'.format(i)] = np.tanh(Z) * dropout / keep_prob
            cache['D{}'.format(i)] = dropout

    return cache
