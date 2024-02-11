#!/usr/bin/env python3
""" This module creates train_mini_batch(X_train, Y_train, X_valid, Y_valid,
batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
save_path="/tmp/model.ckpt"): function"""
import numpy as np


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent

    Inputs:
    X_train - numpy.ndarray of shape (m, 784) containing the training data
        m - number of data points
        784 - number of input features
    Y_train - one-hot numpy.ndarray of shape (m, 10) containing training labels
        10 - number of classes the model should classify
    X_valid - numpy.ndarray of shape (m, 784) containing the validation data
    Y_valid - one-hot numpy.ndarray of shape (m, 10) containing
              the validation labels
    batch_size - number of data points in a batch
    epochs - number of times the training should pass through the whole dataset
    load_path - path from which to load the model
    save_path - path to where the model should be saved after training

    Returns:
    The path where the model was saved
    """
