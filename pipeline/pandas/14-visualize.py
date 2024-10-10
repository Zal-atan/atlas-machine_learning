#!/usr/bin/env python3
""" Module creating the from_dictionary() function"""
import pandas as pd
import matplotlib as plt

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE

# Remove Weighted_Price
df.drop(['Weighted_Price'], axis=1, inplace=True)

# Drop all data before 2017
df = df[(df['Timestamp'] >= 1483228800)]

# Rename Timestamp to Date
df.rename(columns={'Timestamp': 'Date'}, inplace=True)

# Convert Timestamp Values to Date
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Change Index to Date
df.set_index('Date', inplace=True)

# Fill NaN values on close with previous close value
df['Close'] = df['Close'].fillna(method='ffill')

# Fill NaN values on High, Low, Open to Value of Close
df['Open'] = df['Open'].fillna(df['Close'])
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])

# Fill missing values om Volume_(BTC) and Volume_(Currency) with 0'set
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Resample the data to daily intervals and apply the specified aggregations
df_daily = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the daily data
df_daily.plot(subplots=True, figsize=(10, 12))
# plt.tight_layout()
# plt.show()

# Possibly drop Volume_Currency for better view of graph
# df.drop(['Volume_(Currency)'], axis=1, inplace=True)

df_daily.plot()
