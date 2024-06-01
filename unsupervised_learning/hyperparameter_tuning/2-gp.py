#!/usr/bin/env python3
""" This module creates the class GaussianProcess as well as a public
instance method kernel()"""
import numpy as np


class GaussianProcess():
    """
    Represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initializes GaussianProcess

        Inputs:\\
        X_init - numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function\\
        Y_init - numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init\\
        t - number of initial samples\\
        l - length parameter for the kernel\\
        sigma_f - standard deviation given to the output of the
        black-box function\\
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices using
        Radial Basis Function(RBF)

        Inputs:\\
        X1 - numpy.ndarray of shape (m, 1)\\
        X2 - numpy.ndarray of shape (n, 1)\\

        Returns:\\
        Covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """

        # Calculate sum of squares for each element
        X1sum = np.sum(X1**2, 1).reshape(-1, 1)
        X2sum = np.sum(X2**2, 1)

        # Dot product of X1 and X2 transposed
        dot = np.dot(X1, X2.T)

        # Squared distance matrix
        sqdist = X1sum + X2sum - 2 * dot

        # Compute covariance matrix using RBF
        covariance = self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

        return covariance

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a
        Gaussian process

        Inputs:\\
        X_s - numpy.ndarray of shape (s, 1) containing all of the points
        whose mean and standard deviation should be calculated
            s - number of sample points

        Returns: mu, sigma\\
        mu - numpy.ndarray of shape (s,) containing the mean for each point
        in X_s, respectively\\
        sigma - numpy.ndarray of shape (s,) containing the variance for
        each point in X_s, respectively
        """

        # Calculate shape and covariance kernel matrices
        s = X_s.shape[0]
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        # Mean of predictive distribution
        mu = np.dot(np.dot(K_s.T, np.linalg.inv(self.K)), self.Y)
        mu = mu.reshape(s,)

        # Covariance of the predictive distribution
        covar_s = K_ss - np.dot(np.dot(K_s.T, np.linalg.inv(self.K)), K_s)
        sigma = np.diag(covar_s)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process

        Inputs:\\
        X_new is a numpy.ndarray of shape (1,) that represents the
        new sample point\\
        Y_new is a numpy.ndarray of shape (1,) that represents the
        new sample function value

        Updates the public instance attributes X, Y, and K
        """
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
