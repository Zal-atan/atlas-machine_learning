#!/usr/bin/env python3
""" Module creating the from_numpy() function"""
import pandas as pd
import string

def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray:

    Inputs:
    array - np.ndarray

    Returns:
    pd.DataFrame
    """
    # Get numb er of columns
    num_columns = array.shape[1]

    # Create column names from alphabet
    columns = list(string.ascii_uppercase[:num_columns])

    # Create DF
    df = pd.DataFrame(array, columns=columns)

    return df
