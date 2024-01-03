#!/usr/bin/env python3
"""This module creates the cat_matrices2D function, which combines two
2-dimensional matrices based on the given axis."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Takes two matrices as inputs (mat1, mat2) and combines them on the
    optional axis given. Axis is set to 0 by default. Axis = 0 adds the matrix
    as an extra row onto the old matrix, where Axis = 1 adds the the matrix as
    an extra column to each matrix. If the two matrices cannot be concatenated,
    returns None, else returns the concatenated matrix"""

    try:
        new_matrix = []
        if axis == 0:
            if len(mat2[0]) != len(mat1[0]):
                return None
            for matrix in mat1:
                new_matrix.append(matrix.copy())
            for matrix in mat2:
                new_matrix.append(matrix.copy())
        elif axis == 1:
            if len(mat2) > len(mat1):
                return None
            for i in range(len(mat1)):
                new_array = mat1[i].copy()
                for j in range(len(mat2[i])):
                    new_array.append(mat2[i][j])
                new_matrix.append(new_array)
                # print(new_array, new_matrix)
        return new_matrix

    except Exception:
        return None
