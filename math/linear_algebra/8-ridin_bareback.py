#!/usr/bin/env python3
"""This module creates a function for multiplying matrices"""


def mat_mul(mat1, mat2):
    """Multiplies two input 2-Dimensional matrices and returns the result.
    Returns None if two matrices cannot be multiplied"""

    if len(mat1[0]) != len(mat2):
        return None

    return_matrix = []
    for matrix in mat1:
        partial_matrix = []
        for i in range(len(matrix)):
            # print("i={}".format(matrix[i]))
            for j in range(len(mat2[i])):
                # print("j={}".format(mat2[i][j]))
                try:
                    partial_matrix[j] += matrix[i] * mat2[i][j]
                except Exception:
                    partial_matrix.append(matrix[i] * mat2[i][j])
        return_matrix.append(partial_matrix)
    return return_matrix
