#!/usr/bin/env python3
"""This module will make a cofactor() function for calculating the
minor of a matrix"""


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

    multiplier = 1
    determ = 0
    for i in range(len(matrix)):
        leader = matrix[0][i]
        sub_matrix = []

        for row in range(len(matrix)):
            if row == 0:  # Skip the leader row
                continue
            new_row = []

            for col in range(len(matrix)):
                if col == i:  # Skip the leader column
                    continue
                new_row.append(matrix[row][col])

            sub_matrix.append(new_row)
        determ += (multiplier * leader * multi_determinate(sub_matrix))
        multiplier *= -1

    return determ


def determinant(matrix):
    """
    Calculates the determinate of a matrix.

    Input:
    matrix - list of lists whose determinant should be calculated, must be
        square

    Returns:
    The determinant of the matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    for mat in matrix:
        if not isinstance(mat, list):
            raise TypeError("matrix must be a list of lists")
        if len(mat) != len(matrix[0]) or len(mat) != len(matrix):
            raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    return multi_determinate(matrix)


def minor(matrix):
    """
    Calculates the minor of a matrix.

    Input:
    matrix - list of lists whose determinant should be calculated, must be
        square

    Returns:
    The minor of the matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for mat in matrix:
        if not isinstance(mat, list):
            raise TypeError("matrix must be a list of lists")
        if len(mat) != len(matrix[0]) or len(mat) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []

        # Go through matrix position by position, finding the minor for
        # that position by finding the determinate.
        for row_minor in range(len(matrix)):
            minor_row = []
            for column_minor in range(len(matrix)):
                determ_matrix = []

                for row in range(len(matrix)):
                    if row == row_minor:
                        continue
                    new_row = []
                    for column in range(len(matrix)):
                        if column == column_minor:
                            continue
                        new_row.append(matrix[row][column])
                    determ_matrix.append(new_row)

                # Calculate the determinate for each place
                minor_row.append(determinant(determ_matrix))
            minor_matrix.append(minor_row)
        return minor_matrix


def cofactor(matrix):
    """
    Calculates the cofactor of a matrix.

    Input:
    matrix - list of lists whose determinant should be calculated, must be
        square

    Returns:
    The cofactor of the matrix
    """

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for mat in matrix:
        if not isinstance(mat, list):
            raise TypeError("matrix must be a list of lists")
        if len(mat) != len(matrix[0]) or len(mat) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    multiplier = 1
    cofactor_matrix = []
    for row in range(len(matrix)):
        new_row = []
        for column in range(len(matrix)):
            new_row.append((multiplier * matrix[row][column]))
            multiplier *= -1
        cofactor_matrix.append(new_row)

    return minor(cofactor_matrix)
