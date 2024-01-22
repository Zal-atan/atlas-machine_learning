#!/usr/bin/env python3
""" Module creating a class Neuron"""
import numpy as np


class Neuron():
    """ Defines a single neuron"""

    def __init__(self, nx):
        """
        Initiates class Neuron
        nx is the number of input features to the neuron. Must be an integer
        of greater than or equal to 1
        Sets Private attributes W, b, A. W is the weights. b is bias which is
        initially set to 0. A is activated output, default 0.
        """
        nx_is_int = isinstance(nx, int)
        nx_ge_1 = nx >= 1
        if not nx_is_int:
            raise TypeError("nx must be an integer")
        if not nx_ge_1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter function for private attribute W"""
        return self.__W

    @property
    def b(self):
        """Getter function for private attribute b"""
        return self.__b

    @property
    def A(self):
        """Getter function for private attribute A"""
        return self.__A

    def forward_prop(self, X):
        """
        This function calculates the forward propagation of the neuron.
        Starts with a basic matrix multiply for (w * x) + b
        Then inputs the result into the sigmoid function 1/(1 + e^(-result))
        Finally changes self.__a to be the final value and returns this value
        """
        output = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + np.exp(-output))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        C=-1/m(∑(Y⋅log(A)+((1-Y)⋅log(1-A))))
        where m is the number of training examples

        Inputs:
        Y represents the correct labels for input data,
        A represents the activated output of the neuron for each example

        Returns:
        C - Cost of the model
        """
        m = Y.shape[1]
        C = -1 / m * (np.sum((Y * np.log(A)) + ((1 - Y) *
                                                np.log(1.0000001 - A))))
        return C

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.
        Prediction is forward propagation evaluated to a 1 or a 0.

        Inputs:
        X - numpy.ndarray which contains the input data
        Y - numpy.ndarray which contains the correct labels for the input data

        Returns:
        Returns the prediction and the cost of the network a tuple.
        """

        A = self.forward_prop(X)
        predict = np.where(A >= 0.5, 1, 0)

        C = self.cost(Y, A)
        return (predict, C)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neuron
        W = W - (α ⋅ dC/dW)
        dC/dW is the Gradient of loss with respect to W

        b = b - (α * dC/db) # Gradient of the loss with respect to b
        dC/db is the Gradient of the loss with respect to b

        where α is alpha

        .T transposes the matrix to make sure they are the correct shape

        Inputs:
        X - numpy.ndarray which contains the input data
        Y - numpy.ndarray which contains the correct labels for the input data
        A - numpy.ndarray containing activated output for each neuron
        alpha - learning rate

        Output:
        Updates self.__W and self.__B
        """

        m = Y.shape[1]
        dz = A - Y

        dW = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m

        self.__W = self.__W - (alpha * dW).T
        self.__b = self.__b - (alpha * db)
