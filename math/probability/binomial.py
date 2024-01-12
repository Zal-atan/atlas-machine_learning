#!/usr/bin/env python3
"""Module creating a binomial class"""


class Binomial():
    """Creating a class representing a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """ Initializing Binomial class with a value for n and p, either given
        or calculated from data (list)"""
        self.n = n
        self.p = p

        # check if data
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            N = len(data)
            mean = sum(data) / N
            variance = sum(map(lambda x: ((data[x] - mean) ** 2), range(0, N)))
            variance /= N
            p = 1 - (variance / mean)
            n = (sum(data) / p) / N
            self.n = int(round(n))
            self.p = float(mean/self.n)

        if n <= 0:
            raise ValueError("n must be a positive value")
        if not 0 < p < 1:
            raise ValueError("p must be greater than 0 and less than 1")

    def factorial(self, n):
        """Returns the factorial of an input number"""
        if n < 2:
            return 1
        else:
            return n * self.factorial(n-1)

    def pmf(self, k):
        """Calculates the value of the PMD for a given number of 'successes'"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        n_choose_k = (self.factorial(self.n)) / (self.factorial(k) *
                                                 self.factorial(self.n - k))

        return n_choose_k * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of 'successes'"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        return sum(map(lambda x: self.pmf(x), range(0, k+1)))
