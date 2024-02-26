#!/usr/bin/env python3
"""This module creates the train_model function"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
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
    early_stopping - boolean indicates whether early stopping should be used
    patience - patience used for early stopping
    learning_rate_decay - boolean that indicates whether learning rate decay
        should be used
    alpha - initial learning rate
    decay_rate - decay rate
    save_best - boolean indicating whether to save the model after each epoch
        if it is the best
    filepath - file path where the model should be saved

    Return:
    History object generated after training the model
    """
    callback = []

    if save_best:
        callback.append(K.callbacks.ModelCheckpoint(filepath=filepath,
                                                    save_best_only=True))

    if learning_rate_decay and validation_data:
        callback.append(K.callbacks.LearningRateScheduler(
                        lambda epoch: alpha / (1 + decay_rate * epoch),
                        verbose=True))

    if validation_data and early_stopping:
        callback.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                  mode='min',
                                                  patience=patience))

    history = network.fit(data, labels, epochs=epochs,
                          batch_size=batch_size, verbose=verbose,
                          shuffle=shuffle, validation_data=validation_data,
                          callbacks=callback)
    return history
