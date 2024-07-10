#!/usr/bin/env python3
""" Module creating the DRU cell class"""
import numpy as np


class GRUCell():
    """
    Represents a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """
        Class Constructor

        Inputs:
        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs

        Creates: the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
        that represent the weights and biases of the cell
        Wz and bz: for the update gate
        Wr and br: for the reset gate
        Wh and bh: for the intermediate hidden state
        Wy and by: for the output

        """

        # Initialize Weights
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        # Initialize Biases
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """
        Helper function for sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """
        Helper function for softmax
        """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

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

        # Update Gate and Reset Gate
        z = self.sigmoid(np.dot(input_cell, self.Wz) + self.bz)
        r = self.sigmoid(np.dot(input_cell, self.Wr) + self.br)

        # Apply reset gate to the previous hidden state
        input_reset = np.concatenate((r * h_prev, x_t), axis=1)

        # Compute the current hidden state candidate
        h_current = np.tanh(np.dot(input_reset, self.Wh) + self.bh)

        # Compute the next hidden state
        h_next = (1 - z) * h_prev + z * h_current

        # Compute the output
        y = self.softmax((h_next @ self.Wy) + self.by)

        return h_next, y
