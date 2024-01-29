#!/usr/bin/env python3
""" Module creating a class Neuron"""
import numpy as np


class Neuron():
    """ Defines a single neuron"""

    def __init__(self, nx):
        """
        Initiates class Neuron
        nx is the number of input features to the neuron. Must be an integer
        of greater than or equal to 1
        """
        nx_is_int = isinstance(nx, int)
        nx_ge_1 = nx >= 1
        if not nx_is_int:
            raise TypeError("nx must be an integer")
        if not nx_ge_1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
