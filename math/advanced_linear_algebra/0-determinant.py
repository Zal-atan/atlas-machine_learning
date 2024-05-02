#!/usr/bin/env python3
"""This module will make a determinant() function for calculating the
determinant of a matrix"""


def multi_determinate(matrix):
    """
    Calculates the determinate if there are multiple dimensions

    Input:
    Square matrix of at least shape 2x2

    Returns:
    Determinate
    """

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    
    return 1


def determinant(matrix):
    """
    Calculates the determinate of a matrix.

    Input:
    matrix - list of lists whose determinant should be calculated, must be 
        square

    Returns:
    The determinant of the matrix
    """
    
    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    
    if matrix == [[]]:
        return 1
    if len(matrix) == 1 and len(matrix[0]) ==1:
        return matrix[0][0]
    
    for mat in matrix:
        if type(mat) != list:
            raise TypeError("matrix must be a list of lists")
        if len(mat) != len(matrix[0]) or len(mat) != len(matrix):
            raise ValueError("matrix must be a square matrix")

    return multi_determinate(matrix)
