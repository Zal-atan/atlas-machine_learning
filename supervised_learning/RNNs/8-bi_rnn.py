#!/usr/bin/env python3
""" Module creating the bi_rnn function"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a deep RNN

    Inputs:
    bi_cell: instance of BidirectinalCell that will be used for the
    forward propagation
    X: data to be used, given as a numpy.ndarray of shape (t, m, i)
        t: maximum number of time steps
        m: batch size
        i: dimensionality of the data
    h_0: initial hidden state, given as a numpy.ndarray of shape (m, h)
        h: dimensionality of the hidden state
    h_t; initial hidden state in the backward direction, given as a
    numpy.ndarray of shape (m, h)

    Returns: H, Y
    H: numpy.ndarray containing all of the hidden states
    Y: numpy.ndarray containing all of the outputs
    """

    t, m, i = X.shape
    h = h_0.shape[1]
    time_step = range(t)

    # Initialize hidden states for forward and backward directions
    H_ford = np.zeros((t + 1, m, h))
    H_back = np.zeros((t + 1, m, h))

    # Set initial hidden states
    H_ford[0] = h_0
    H_back[t] = h_t

    # Reverse the input data for the backward pass
    X_back = np.flip(X, 0)

    # Perform forward and backward passes
    for t in time_step:
        H_ford[t + 1] = bi_cell.forward(H_ford[t], X[t])
        H_back[t + 1] = bi_cell.backward(H_back[t], X_back[t])

    # Concatenate forward and backward hidden states
    H = np.concatenate((H_ford[1:], H_back[7:0:-1]), axis=2)

    # Compute the output using the bidirectional hidden states
    Y = bi_cell.output(H)

    return H, Y
