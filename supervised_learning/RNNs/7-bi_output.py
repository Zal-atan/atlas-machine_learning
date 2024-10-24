#!/usr/bin/env python3
""" Module creating the BidirectionalCell class"""
import numpy as np


class BidirectionalCell():
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

        Creates: the public instance attributes Whf, Whb, Wy, bhf, bhb, by
        that represent the weights and biases of the cell
        Whf and bhf: for the hidden states in the forward direction
        Whb and bhb: for the hidden states in the backward direction
        Wy and by: for the outputs
        """

        # Initialize Weights
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))

        # Initialize Biases
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
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

        Returns: h_next
        h_next: next hidden state
        """

        # Previos hidden layer and input data
        input_cell = np.concatenate((h_prev, x_t), axis=1)

        # Next hidden state
        h_next = np.tanh(np.matmul(input_cell, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time step

        Inputs:
        x_t: numpy.ndarray of shape (m, i) that contains the data input
        for the cell
            m: batch size for the data
        h_next: numpy.ndarray of shape (m, h) containing the next hidden state

        Return:
        h_prev: the previous hidden state
        """
        # next hidden layer and input data
        input_cell = np.concatenate((h_next, x_t), axis=1)

        h_prev = np.tanh(np.matmul(input_cell, self.Whb) + self.bhb)

        return h_prev

    def output(self, H):
        """
        Calculates all outputs for the RNN

        Input:
        H: numpy.ndarray of shape (t, m, 2 * h) that contains the concatenated
        hidden states from both directions, excluding their initialized states
            t: number of time steps
            m: batch size for the data
            h: dimensionality of the hidden states

        Return:
        Y: the outputs
        """

        Y = np.dot(H, self.Wy) + self.by
        Y = np.exp(Y) / np.sum(np.exp(Y), axis=2, keepdims=True)

        return Y
