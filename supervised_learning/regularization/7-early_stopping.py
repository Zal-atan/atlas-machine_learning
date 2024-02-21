#!/usr/bin/env python3
"""This module will create the early_stopping function"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early

    Inputs:
    cost - current validation cost of the neural network
    opt_cost - lowest recorded validation cost of the neural network
    threshold - threshold used for early stopping
    patience - patience count used for early stopping
    count - count of how long the threshold has not been met

    Returns:
    Boolean of whether the network should be stopped early, followed by the updated count
    """

    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    if count < patience:
        return False, count
    else:
        return True, count
