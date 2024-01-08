#!/usr/bin/env python3
"""Create function poly_derivative(poly), which will return the derivative
of a list of polynomials."""


def poly_derivative(poly):
    """ Returns a list of derivatives of polynomials, input is a list.
    Example: x^3 + 3x +5 would be input as [5, 3, 0, 1]
    Output would be: [3, 0, 3]"""

    if isinstance(poly, list) and len(poly) > 0:
        if len(poly) == 1:
            return [0]
        for value in poly:
            if not isinstance(value, int):
                return None
        return_list = []
        for i in range(1, len(poly)):
            return_list.append(i * poly[i])

        return return_list

    else:
        return None

    # if not isinstance(poly, list):
    #     return None

    # for value in poly:
    #     if not isinstance(value, int):
    #         return None

    # if len(poly) == 1:
    #     return [0]

    # return_list = []
    # for i in range(1, len(poly)):
    #     return_list.append(i * poly[i])

    # return return_list
