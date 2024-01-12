#!/usr/bin/env python3
"""Module for creating class Poisson"""
e = 2.7182818285


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
        """Returns the factorial of an input number"""
        if n < 2:
            return 1
        else:
            return n * self.factorial(n-1)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of 'successes'"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        pmf = (((e ** (-self.lambtha)) * (self.lambtha ** k))
               / self.factorial(k))
        return pmf

    def cdf(self, k):
        """Calculates the value of CDF for a given number of 'successes'"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        return sum(map(lambda x: (((e ** -self.lambtha) * (self.lambtha ** x))
                                  / self.factorial(x)), range(0, k + 1)))
