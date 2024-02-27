#!/usr/bin/env python3
"""This module creates the save_model and load_model functions"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model

    Inputs:
    network - model to save
    filename - path of the file that the model should be saved to

    Return:
    None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model

    Inputs:
    filename - path of the file that the model should be saved to

    Return:
    The loaded model
    """
    load = K.models.load_model(filename)
    return load
