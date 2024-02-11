#!/usr/bin/env python3
""" This module creates update_variables_RMSProp(alpha, beta2, epsilon,
var, grad, s): function"""
import tensorflow.compat.v1 as tf


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using the RMSProp optimization algorithm

    Inputs:
    alpha - learning rate
    beta2 - RMSProp weight
    epsilon - small number to avoid division by zero
    var - numpy.ndarray containing the variable to be updated
    grad - numpy.ndarray containing the gradient of var
    s - previous second moment of var

    Returns:
    The updated variable and the new moment, respectively
    """
    moment = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    updated_var = var - (alpha * (grad / ((moment ** (1 / 2)) + epsilon)))

    return updated_var, moment
