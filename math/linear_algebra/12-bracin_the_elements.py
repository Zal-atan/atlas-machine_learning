#!/usr/bin/env python3
"""Creates a function which returns add, subtract, multiplu, and divide
of two matrices, all inside a tuple"""


def np_elementwise(mat1, mat2):
    """Returns a tuple of each of the different operations results (add,
    subtract, multiply, and divide) of two matrices"""
    return ((mat1 + mat2), (mat1 - mat2), (mat1 * mat2), (mat1 / mat2))
