#!/usr/bin/env python3
""" This module creates update_variables_Adam(alpha, beta1, beta2, epsilon,
var, grad, v, s, t) function
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Creates the training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm

    Inputs:
    alpha - learning rate
    beta1 - weight used for the first moment
    beta2 - weight used for the second moment
    epsilon - small number to avoid division by zero
    var - numpy.ndarray containing the variable to be updated
    grad - numpy.ndarray containing the gradient of var
    v - previous first moment of var
    s - previous second moment of var
    t - time step used for bias correction

    Returns:
    The updated variable, the new first moment, and the new second moment,
    respectively
    """
    V = (beta1 * v) + ((1 - beta1) * grad)
    S = (beta2 * s) + ((1 - beta2) * (grad ** 2))

    V_corrected = V / (1 - (beta1 ** t))
    S_corrected = S / (1 - (beta2 ** t))

    updated_var = var - alpha * (V_corrected / ((S_corrected ** (1 / 2))
                                                + epsilon))

    return updated_var, V_corrected, S_corrected
