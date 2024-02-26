#!/usr/bin/env python3
"""This module creates the one_hot function"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix:

    Inputs:
    labels - labels for the one hot

    Return:
    The one-hot matrix
    """
    one_hot = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot
