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
        Sets Private attributes W, b, A. W is the weights. b is bias which is
        initially set to 0. A is activated output, default 0.
        """
        nx_is_int = isinstance(nx, int)
        nx_ge_1 = nx >= 1
        if not nx_is_int:
            raise TypeError("nx must be an integer")
        if not nx_ge_1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter function for private attribute W"""
        return self.__W

    @property
    def b(self):
        """Getter function for private attribute b"""
        return self.__b

    @property
    def A(self):
        """Getter function for private attribute A"""
        return self.__A
