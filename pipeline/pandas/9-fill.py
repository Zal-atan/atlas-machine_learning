#!/usr/bin/env python3
""" Module creating the from_dictionary() function"""
import pandas as pd

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE

# Drop Column Weighted_Price
df.drop(columns=['Weighted_Price'])

# Fill NaN values on close with previous close value
df['Close'] = df['Close'].fillna(method='ffill')

# Fill NaN values on High, Low, Open to Value of Close
# df[['High', 'Low', 'Open']] = df[['High', 'Low', 'Open']].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])

# Fill missing values om Volume_(BTC) and Volume_(Currency) with 0'set
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

print(df.head())
print(df.tail())
