#!/usr/bin/env python3
""" This file is for preprocessing the data for a bitcoin forecaster."""

import pandas as pd
import numpy as np


def preprocess():
    """
    Takes input coinbase data, and preprocess it using Pandas.

    Returns:
    train_data: normalized and cleaned training data 70%
    valid_data: normalized and cleaned validation data 20%
    test_data: normalized and cleaned testingdata 10%
    """
    coinbase_data = "./data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"

    try:
        coinbase_df = pd.read_csv(coinbase_data)
    except:
        raise ImportError("File path does not exist.")

    coinbase_df = coinbase_df[coinbase_df['Timestamp'] >= 1500000000]

    # Linear interpolation to fill fill in values for NaN's
    coinbase_df = coinbase_df.interpolate()

    # Plot again to see
    coinbase_df.plot(x='Timestamp', y=['Close'], kind='line')

    # Hour intervals of dataset
    df = coinbase_df[::60]

    n = len(df)
    train_data = df[0:int(n*.7)]
    valid_data = df[int(n*.7):int(n*.9)]
    test_data = df[int(n*.9):]

    # Normalize the data

    train_mean = train_data.mean()
    train_std = train_data.std()

    train_data = (train_data - train_mean) / train_std
    valid_data = (valid_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    return train_data, valid_data, test_data
