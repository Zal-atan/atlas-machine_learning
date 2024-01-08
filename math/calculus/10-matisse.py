#!/usr/bin/env python3
"""Create function poly_derivative(poly), which will return the derivative
of a list of polynomials."""

def poly_derivative(poly):
    """ Returns a list of derivatives of polynomials, input is a list.
    Example: x^3 + 3x +5 would be input as [5, 3, 0, 1]
    Output would be: [3, 0, 3]"""

    return_list = []
    for value in poly:
        if not isinstance(value, int):
            return None
    for i in range(1, len(poly)):
        return_list.append(i * poly[i])

    return return_list
