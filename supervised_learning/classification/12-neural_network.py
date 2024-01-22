#!/usr/bin/env python3
""" Module creating a class NeuronalNetwork"""
import numpy as np


class NeuralNetwork():
    """ Defines a neural network with one hidden layer"""

    def __init__(self, nx, nodes):
        """
        Initiates Neural Network class

        Inputs:
        nx - number of input features
            * must be integer of value greater than or equal to 1
        nodes - number of nodes found in the hidden layer
            * must be integer of value greater than or equal to 1

        Private Instance Attributes:
        W1 - weights vector for the hidden layer, initiated with a random
            normal distribution
        b1 - bias for the hidden layer, initiated with 0's
        A1 - activated output for the hidden layer, initiated with 0
        W2 - weights vector for the output neuron, initiated with a random
            normal distribution
        b2 - bias for the output neuron, initiated with 0's
        A2 - activated output for the output neuron, initiated with 0
        """

        nx_is_int = isinstance(nx, int)
        nx_ge_1 = nx >= 1
        nodes_is_int = isinstance(nodes, int)
        nodes_ge_1 = nodes >= 1

        if not nx_is_int:
            raise TypeError("nx must be an integer")
        if not nx_ge_1:
            raise ValueError("nx must be a positive integer")
        if not nodes_is_int:
            raise TypeError("nodes must be an integer")
        if not nodes_ge_1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter for the private W1 attribute
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter for the private b1 attribute
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter for the private A1 attribute
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter for the private W2 attribute
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter for the private b2 attribute
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter for the private A2 attribute
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.
        Starts with a basic matrix multiply for (w * x) + b
        Then inputs the result into the sigmoid function 1/(1 + e^(-result))

        Inputs:
        X - numpy.ndarray which contains the input data

        Returns:
        Updates and returns __A1 and __A2
        """
        output1 = np.matmul(self.__W1, X) + self.__b1
        sigmoid1 = 1 / (1 + np.exp(-output1))
        self.__A1 = sigmoid1

        output2 = np.matmul(self.__W2, self.__A1) + self.__b2
        sigmoid2 = 1 / (1 + np.exp(-output2))
        self.__A2 = sigmoid2

        return (self.__A1, self.__A2)

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

        A, B = self.forward_prop(X)
        predict = np.where(B >= 0.5, 1, 0)

        C = self.cost(Y, B)
        return (predict, C)
