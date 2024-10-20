#!/usr/bin/env python3
""" Module creating the from_dictionary() function"""
import pandas as pd

from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# YOUR CODE HERE

# Set indexes to timestamps
df1 = df1.set_index('Timestamp')
df2 = df2.set_index('Timestamp')

# # Filter df2 and df1
df2 = df2.loc[1417411980:1417417980]
df1 = df1.loc[1417411980:1417417980]

# Concatenate
df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

# sort
df = df.swaplevel(0, 1).sort_index()

print(df)
