#!/usr/bin/env python3
""" This module creates moving_average(data, beta): function"""
import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    Inputs:
    data - list of data to calculate the moving average of
    beta - weight used for the moving average

    Returns:
    List of the moving averages of data

    Formula:
    EMA = (beta * prev_weighted_avg) + ((1- beta) * data[data_number])
    bias = EMA / (1 - (beta ** data_number))
    """

    EMA = 0
    EMA_list = []
    for i in range(len(data)):
        EMA = (beta * EMA) + ((1 - beta) * data[i])
        bias_correction = EMA / (1 - (beta ** (i + 1)))
        EMA_list.append(bias_correction)

    return EMA_list
