#!/usr/bin/env python3
"""Module creating a normal class"""
pi = 3.1415926536
e = 2.7182818285


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

    def z_score(self, x):
        """Calculates a z score of a given x _value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        return ((1 / (self.stddev * ((2 * pi) ** .5))) *
                (e ** (-1 / 2 * ((self.z_score(x)) ** 2))))

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""

        # Calculating part inside error function
        inside = (x - self.mean) / (self.stddev * (2 ** .5))

        # Calculating error function basically using Taylor Series expansion
        error_func = (2 / (pi ** .5)) * (inside - ((inside **3) / 3) +
                                         ((inside ** 5) / 10) -
                                         ((inside ** 7) / 42) +
                                         ((inside ** 9) / 216))

        return (1 + error_func) / 2
