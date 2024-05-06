#!/usr/bin/env python3
""" This module creates the Multinormal Class"""
import numpy as np


class MultiNormal:
    """ Represents the Multivariate Normal Distribution"""

    def __init__(self, data):
        """
        Initializes Class

        Input:
        data - numpy.ndarray of shape (d, n) containing the data set
            n - number of data points
            d  number of dimensions in each data point
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        self.cov = np.matmul(data - self.mean, data.T - self.mean.T) / (n - 1)

    def pdf(self, x):
        """
        Public instance method which calculates the PDF at a data point

        Input:
        x - numpy.ndarray of shape (d, 1) containing the data point whose
        PDF should be calculated
            d - number of dimensions of the Multinomial instance
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError(f"x must have the shape ({d}, 1)")

        if x.shape[0] != d or x.shape[1] != 1:
            raise ValueError(f"x must have the shape ({d}, 1)")

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        coef = 1 / np.sqrt((2 * np.pi) ** d * det)
        exponent = -0.5 * np.dot((x - self.mean).T,
                                 np.dot(inv, (x - self.mean)))

        return float(coef * np.exp(exponent))
