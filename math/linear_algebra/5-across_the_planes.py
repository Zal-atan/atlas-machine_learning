#!/usr/bin/env python3
"""This module creates the function add_matrices2D for adding two dimensional
matrices"""

def add_matrices2D(mat1, mat2):
    """This function takes input of two matrices (mat1, mat2), adds the two
    matrices element wise, and returns the resultant matrix. Returns none if
    the two matrices are of different shape"""

    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None

    resultant_matrix = []
    for i in range(len(mat1)):
        new_element = []
        for j in range(len(mat1[i])):
            new_element.append(mat1[i][j] + mat2[i][j])
        resultant_matrix.append(new_element)
    return resultant_matrix
