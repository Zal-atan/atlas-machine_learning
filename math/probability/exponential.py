#!/usr/bin/env python3
"""Module for creating Exponential class"""


class Exponential():
    """Creates Exponential class representing an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initializes Exponential class with a some data (list) and a
        value for lambtha (either calculated from data or given)"""
        # initialize lambtha
        if not lambtha > 0:
            raise ValueError("lambtha must be a positive value")
        self.lambtha = lambtha

        # check if data
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))
