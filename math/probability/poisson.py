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

    def factorial(self, n):
        if n < 2:
            return 1
        else:
            return n * self.factorial(n-1)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of "successes"""
        if not isinstance(k, int):
            k = int(k)
        e =  2.718281828459045
        pmf = (((e ** (-self.lambtha)) * (self.lambtha ** k))
               / self.factorial(k))
        return pmf
