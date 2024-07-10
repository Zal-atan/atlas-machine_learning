#!/usr/bin/env python3
""" Module creating the rnn function"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN

    Inputs:
    rnn_cell: instance of RNNCell that will be used for the forward propagation
    X: data to be used, given as a numpy.ndarray of shape (t, m, i)
        t: maximum number of time steps
        m: batch size
        i: dimensionality of the data
    h_0: initial hidden state, given as a numpy.ndarray of shape (m, h)
        h: dimensionality of the hidden state

    Returns: H, Y
    H: numpy.ndarray containing all of the hidden states
    Y: numpy.ndarray containing all of the outputs
    """

    t, m, i = X.shape
    h = h_0.shape[1]
    time_step = range(t)

    # Initialize hidden states of H
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # Determine output dimension from rnn_cell
    o = rnn_cell.Wy.shape[1]

    # Initialize outputs array Y
    Y = np.zeros((t, m, o))

    # Loop through each time step to compute hidden states and outputs
    for t in time_step:
        H[t + 1], Y[t] = rnn_cell.forward(H[t], X[t])

    return H, Y
