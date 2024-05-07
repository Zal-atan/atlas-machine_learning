#!/usr/bin/env python3
""" This module defines the posterior function"""
import numpy as np
from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that the probability of developing
    severe side effects falls within a specific range given the data

    Inputs:
    x - number of patients that develop severe side effects
    n - total number of patients observed
    p1 - lower bound on the range
    p2 - upper bound on the range

    Return:
    posterior probability of each probability in P given x and n, respectively
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or not 0 <= p1 <= 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or not 0 <= p2 <= 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    beta_distrib1 = special.btdtr(x + 1, n - x + 1, p1)
    beta_distrib2 = special.btdtr(x + 1, n - x + 1, p2)

    posterior = beta_distrib2 - beta_distrib1

    return posterior
