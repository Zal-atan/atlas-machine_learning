#!/usr/bin/env python3
""" Module creating the LSTMcell class"""
import numpy as np


class LSTMCell():
    """
    Represents an LSTM unit
    """

    def __init__(self, i, h, o):
        """
        Class Constructor

        Inputs:
        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs

        Creates: the public instance attributes Wf, Wu, Wc, Wo, Wy,
        bf, bu, bc, bo, by that represent the weights and biases of the cell
        Wf and bf: for the forget gate
        Wu and bu: for the update gate
        Wc and bc: for the intermediate cell state
        Wo and bo: for the output gate
        Wy and by: for the outputs

        """

        # Initialize Weights
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        # Initialize Biases
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """
        Public Instance method which performs forward propagation for
        one time step

        Inputs:
        x_t: numpy.ndarray of shape (m, i) that contains the data input for
        the cell
            m:  the batch size for the data
        h_prev: numpy.ndarray of shape (m, h) containing the previous
        hidden state
        c_prev: numpy.ndarray of shape (m, h) containing the
        previous cell state

        Returns: h_next, c_next, y
        h_next: next hidden state
        c_next: next cell state
        y: output of the cell
        """

        # Previos hidden layer and input data
        input_cell = np.concatenate((h_prev, x_t), axis=1)

        # Forget Gate, Update Gate, Output Gate
        f = self.sigmoid(np.dot(input_cell, self.Wf) + self.bf)
        u = self.sigmoid(np.dot(input_cell, self.Wu) + self.bu)
        o = self.sigmoid(np.dot(input_cell, self.Wo) + self.bo)

        # Compute the current candidate c state
        c_current = np.tanh(np.dot(input_cell, self.Wc) + self.bc)

        # Next candidate cell
        c_next = c_prev * f + u * c_current

        # Compute the next hidden state
        h_next = o * np.tanh(c_next)

        # Compute the output
        y = self.softmax((h_next @ self.Wy) + self.by)

        return h_next, c_next, y
