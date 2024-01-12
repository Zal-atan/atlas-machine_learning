#!/usr/bin/env python3
"""Module creating a normal class"""


class Normal():
    """Creating a class representing a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initializes Normal class with a value of mean and stddev either
        given or produced from the input list (data)"""
        self.mean = mean
        self.stddev = stddev

        # check if data
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            n = len(data)
            self.mean = sum(data) / n
            self.stddev = (1/n * sum(map(lambda x:
                                         ((data[x] - self.mean) ** 2),
                                         range(0, n)))) ** .5

        if self.stddev <= 0:
            raise ValueError("stddev must be a positive value")
