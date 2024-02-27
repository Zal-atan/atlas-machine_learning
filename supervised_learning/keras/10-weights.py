#!/usr/bin/env python3
"""This module creates the save_weights and load_weights functions"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Saves a model's weights

    Inputs:
    network - model to save
    filename - path of the file that the model should be saved to
    save_format - format in which the weights should be saved

    Return:
    None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Loads an entire model

    Inputs:
    network - model to which the weights should be loaded
    filename - path of the file that the model should be saved to

    Return:
    None
    """
    network.load_weights(filename)
    return None
