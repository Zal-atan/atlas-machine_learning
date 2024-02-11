#!/usr/bin/env python3
""" This module creates update_variables_momentum(alpha, beta1, var, grad, v):
function"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization
    algorithm

    Inputs:
    alpha - learning rate
    beta1 - momentum weight
    var - numpy.ndarray containing the variable to be updated
    grad - numpy.ndarray containing the gradient of var
    v - previous first moment of var

    Returns:
    The updated variable and the new moment, respectively
    """

    EMA = 0
    EMA_list = []
    for i in range(len(data)):
        EMA = (beta * EMA) + ((1 - beta) * data[i])
        bias_correction = EMA / (1 - (beta ** (i + 1)))
        EMA_list.append(bias_correction)

    return EMA_list
