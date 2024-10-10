#!/usr/bin/env python3
""" Module creating the from_dictionary() function"""
import pandas as pd


# Create dict
dictionary = {
    'A': [0.0, 'one'],
    'B': [0.5, 'two'],
    'C': [1.0, 'three'],
    'D': [1.5, 'four']
}

# Column Names
columns = ['First', 'Second']

df = pd.DataFrame.from_dict(dictionary, orient='index', columns=columns)
