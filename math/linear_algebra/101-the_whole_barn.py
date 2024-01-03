#!/usr/bin/env python3
"""This module adds two matrices"""


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


def add_matrices(mat1, mat2):
    """Function for adding mat1 to mat2, returns resultant matrix"""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if shape1 != shape2:
        return None

    new_matrix = []
    for i in range(shape1[0]):
        first_tier = []
        try:
            for j in range(shape1[1]):
                second_tier = []
                try:
                    for k in range(shape1[2]):
                        third_tier = []
                        try:
                            for last in range(shape1[3]):
                                third_tier.append(mat1[i][j][k][last] +
                                                  mat2[i][j][k][last])
                        except Exception:
                            second_tier.append(mat1[i][j][k] + mat2[i][j][k])
                        else:
                            second_tier.append(third_tier)
                except Exception:
                    first_tier.append(mat1[i][j] + mat2[i][j])
                else:
                    first_tier.append(second_tier)
        except Exception:
            new_matrix.append(mat1[i] + mat2[i])
        else:
            new_matrix.append(first_tier)
    return new_matrix
