#!/usr/bin/env python3
"""Creates a function which returns add, subtract, multiplu, and divide
of two matrices, all inside a tuple"""
import numpy as np


def np_elementwise(mat1, mat2):
    """Returns a tuple of each of the different operations results (add,
    subtract, multiply, and divide) of two matrices"""
    return ((np.add(mat1, mat2)), (np.subtract(mat1, mat2)),
            (np.multiply(mat1, mat2)), (np.divide(mat1, mat2)))
