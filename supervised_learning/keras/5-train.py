#!/usr/bin/env python3
"""This module creates the train_model function"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent:

    Inputs:
    network - model to train
    data - numpy.ndarray of shape (m, nx) containing the input data
    labels - one-hot numpy.ndarray of shape (m, classes)
             containing the labels of data
    batch_size - size of the batch used for mini-batch gradient descent
    epochs - number of passes through data for mini-batch gradient descent
    validation_data - data to validate the model with, if not None
    verbose - boolean determines if output should be printed during training
    shuffle - boolean determines whether to shuffle the batches every epoch.
              Normally, it is a good idea to shuffle, but for reproducibility,
              we have chosen to set the default to False.

    Return:
    History object generated after training the model
    """
    history = network.fit(data, labels, epochs=epochs,
                          batch_size=batch_size, verbose=verbose,
                          shuffle=shuffle, validation_data=validation_data)
    return history
