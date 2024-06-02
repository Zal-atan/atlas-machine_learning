#!/usr/bin/env python3
""" This module creates the class BayesianOptimization"""
from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Initializes BayesianOptimization

        Inputs:\\
        f - black-box function to be optimized\\
        X_init - numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function\\
        Y_init - numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init\\
        t - number of initial samples\\
        bounds - tuple of (min, max) representing the bounds of the
        space in which to look for the optimal point\\
        ac_samples - number of samples that should be analyzed
        during acquisition\\
        l - length parameter for the kernel\\
        sigma_f - standard deviation given to the output of the
        black-box function\\
        xsi - exploration-exploitation factor for acquisition\\
        minimize - bool determining whether optimization should be
        performed for minimization (True) or maximization (False)

        Public Instance Attributes:
        f: the black-box function
        gp: an instance of the class GaussianProcess
        X_s: a numpy.ndarray of shape (ac_samples, 1) containing all
        acquisition sample points, evenly spaced between min and max
        xsi: the exploration-exploitation factor
        minimize: a bool for minimization versus maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        start, stop = bounds
        X_s = np.linspace(start, stop, ac_samples)
        self.X_s = (np.sort(X_s)).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location:
        Uses the Expected Improvement acquisition function

        Returns: X_next, EI\\
        X_next - numpy.ndarray of shape (1,) representing the next
        best sample point\\
        EI - numpy.ndarray of shape (ac_samples,) containing the
        expected improvement of each potential sample\\
        """

        # Predict the mean and standard deviation for training points and
        # sample points
        mu, sigma = self.gp.predict(self.gp.X)
        mu_s, sigma_s = self.gp.predict(self.X_s)

        # Determine current best mean value
        mu_bound = np.max(mu)
        if self.minimize:
            mu_bound = np.min(mu)

        # Calculate improvement factor
        num_Z = mu_bound - mu_s - self.xsi
        Z = num_Z / sigma_s

        # Calculate Expected Improvement for each sample point
        EI = np.array(num_Z * norm.cdf(Z) + sigma_s * norm.pdf(Z))

        # Determine index of max EI sample point and find X_next
        index = np.argmax(EI)
        X_next = self.X_s[index]

        return X_next, EI
