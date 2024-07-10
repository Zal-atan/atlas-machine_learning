#!/usr/bin/env python3
""" Module creating the deep_rnn function"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    Inputs:
    rnn_cells: instance of RNNCell that will be used for the forward
    propagation
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
    h = h_0.shape[2]
    time_step = range(t)
    layers = len(rnn_cells)

    # Initialize hidden states of H
    H = np.zeros((t + 1, layers, m, h))
    H[0] = h_0

    # Determine output dimension from rnn_cell
    o = rnn_cells[-1].Wy.shape[1]

    # Initialize outputs array Y
    Y = np.zeros((t, m, o))

    # Loop through each time step to compute hidden states and outputs
    for t in time_step:
        x_t = X[t]
        for lay in range(layers):
            rnn_cell = rnn_cells[lay]
            h_prev = H[t, lay]
            h_next, Y[t] = rnn_cell.forward(h_prev, x_t)
            H[t + 1, lay] = h_next
            # Propagate the hidden state to the next layer
            x_t = h_next

    return H, Y
