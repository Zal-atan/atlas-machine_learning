#!/usr/bin/env python3
"""This module will create the l2_reg_gradient_descent function"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization

    Inputs:
    Y - one-hot numpy.ndarray of shape (classes, m) that
        contains the correct labels for the data
            classes - number of classes
            m - number of data points
    weights - dictionary of the weights and biases of the neural network
    cache - dictionary of the outputs of each layer of the neural network
    alpha - learning rate
    lambtha - L2 regularization parameter
    L - number of layers of the network
    """
    m = (1 / Y.shape[1])
    dZ = cache['A{}'.format(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A{}'.format(i - 1)]
        W_current = weights['W{}'.format(i)]

        L2_cost = m * lambtha * W_current
        dW = m * np.matmul(dZ, A_prev.T) + L2_cost
        db = m * np.sum(dZ, axis=1, keepdims=True)

        dZ = np.matmul(W_current.T, dZ) * (1 - np.power(A_prev, 2))

        weights['W{}'.format(i)] -= (alpha *dW)
        weights['b{}'.format(i)] -= (alpha * db)
