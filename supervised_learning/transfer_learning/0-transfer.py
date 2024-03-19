#!/usr/bin/env python3
"""Module which trains a convolutional neural network to classify the 
CIFAR 10 dataset"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for your model

    Inputs:
    X - numpy.ndarray (m, 32, 32, 3) containing the CIFAR 10 data, where
        m is the number of data points
    Y - numpy.ndarray (m,) containing the CIFAR 10 labels for X

    Returns:
    X_p - numpy.ndarray containing the preprocessed X
    Y_p - numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

