#!/usr/bin/env python3
""" Module creating the from_dictionary() function"""
import pandas as pd

from_file = __import__('2-from_file').from_file

df_task3 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE

# Change Timestamp to Datetime
df_task3.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
df_task3['Datetime'] = pd.to_datetime(df_task3['Datetime'], unit='s')

# Remove all columns except Datetime and Close
df_task3 = df_task3[['Datetime', 'Close']]

print(df_task3.tail())
