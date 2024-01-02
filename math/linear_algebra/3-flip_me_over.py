#!/usr/bin/env python3
""" This module makes function matrix_transpose"""

def matrix_transpose(matrix):
    """Takes a matrix and returns a transposed matrix"""
    transposed = []
    i = 0
    while i < len(matrix[0]):
        new_matrix = []
        for mat in matrix:
            new_matrix.append(mat[i])
        transposed.append(new_matrix)
        i += 1
    return transposed
