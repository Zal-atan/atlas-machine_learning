#!/usr/bin/env python3
""" Module creating a class NeuronalNetwork"""
import numpy as np


class DeepNeuralNetwork():
    """ Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Initiates  Deep Neural Network class

        Inputs:
        nx - number of input features
            * must be integer of value greater than or equal to 1
        layers - number of nodes found in the each layer of the network
            * must be a list of positive integers

        Public Instance Attributes:
        L - The number of layers in the neural network
        cache - A dictionary holding all intermediary values of the network.
            Empty on instantiation
        weights - Dictionary holding all weights and biases of the network
        """

        nx_is_int = isinstance(nx, int)
        nx_ge_1 = nx >= 1
        layers_is_list_ints = isinstance(layers, list)
        if len(layers) < 1:
            layers_is_list_ints = False

        if not nx_is_int:
            raise TypeError("nx must be an integer")
        if not nx_ge_1:
            raise ValueError("nx must be a positive integer")
        if not layers_is_list_ints:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        previous = nx
        weights_dict = {}

        for l in range(self.L):
            if not isinstance(layers[l], int) or layers[l] < 0:
                raise TypeError("layers must be a list of positive integers")

            weights_dict[f"b{l}"] = np.zeros((layers[l], 1))
            weights_dict[f"W{l}"] = (np.random.randn(layers[l], previous) *
                                     np.sqrt(2/ previous))
            previous = layers[l]

        self.weights = weights_dict
