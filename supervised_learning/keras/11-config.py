#!/usr/bin/env python3
"""This module creates the save_config and load_config functions"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's weights

    Inputs:
    network - model to save
    filename - path of the file that the model should be saved to

    Return:
    None
    """
    with open(filename, "w") as file:
        file.write(network.to_json())

    return None


def load_config(filename):
    """
    Loads an entire model

    Inputs:
    filename - path of the file that the model should be saved to

    Return:
    The loaded model
    """
    with open(filename, "r") as file:
        network = file.read()

    return K.models.model_from_json(network)
