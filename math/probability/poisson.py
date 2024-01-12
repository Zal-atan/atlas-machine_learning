#!/usr/bin/env python3
"""Module for creating class Poisson"""


class Poisson():
    """Creates Poisson class"""

    def __init__(self, data=None, lambtha=1.):
        """Initializes Poisson class with data (list) values and a lambtha"""
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
            self.lambtha = sum(data) / len(data)
