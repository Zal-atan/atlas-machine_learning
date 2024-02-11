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
    V = (beta1 * v) + ((1 - beta1) * grad)
    W = var - (alpha * V)
    return W, V
