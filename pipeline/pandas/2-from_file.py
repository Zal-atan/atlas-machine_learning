#!/usr/bin/env python3
""" Module creating the from_dictionary() function"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame

    Inputs:
    filename - filename/location
    delimiter - type of delimiter between information, ex. commas

    Returns:
    loaded pd.DataFrame
    """
    df = pd.read_csv(filename,delimiter=delimiter)

    return df
