#!/usr/bin/env python3
""" Module creating the from_dictionary() function"""
import pandas as pd

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE

# Flip to reverse chronological
df = df.sort_values(by=['Timestamp'], ascending=False)

# Transpose rows and columns
df = df.transpose()

print(df.tail(8))
