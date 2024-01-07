#!/usr/bin/env python3
"""This module makes a function which can find the size of matrix, given that
all the elements of the same dimension are of the same type/shape"""


def matrix_shape(matrix):
    """Takes in a matrix of equal type/shape in each dimension and returns
    the shape of the matrix"""
    shape = []
    shape.append(len(matrix))
    new_matrix = matrix
    while True:
        try:
            new_matrix = new_matrix[0]
            shape.append(len(new_matrix))
        except Exception:
            break
    return shape
