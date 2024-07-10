#!/usr/bin/env python3
""" Module creating the RNN class"""
import numpy as np


class RNNCell():
    """
    Represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        Class Constructor

        Inputs:
        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs

        Creates the public instance attributes Wh, Wy, bh, by that represent
        the weights and biases of the cell
        Wh and bh: for the concatenated hidden state and input data
        Wy and by: for the output
        """

        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Public Instance method which performs forward propagation for
        one time step

        Inputs:
        x_t: numpy.ndarray of shape (m, i) that contains the data input for
        the cell
            m:  the batch size for the data
        h_prev: numpy.ndarray of shape (m, h) containing the previous
        hidden state

        Returns:
        h_next: next hidden state
        y: output of the cell
        """

        # Previos hidden layer and input data
        input_cell = np.concatenate((h_prev, x_t), axis=1)

        # Next hidden state
        h_next = np.tanh(np.matmul(input_cell, self.Wh) + self.bh)

        # Compute
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
