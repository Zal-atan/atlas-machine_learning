#!/usr/bin/env python3
""" Creates function poly_integral(poly, C=0):, which returns a list of
the integrated polynomials"""


def format_number(n):
    """Formats the number to be an integer if possible, else a float"""
    if n == int(n):
        return int(n)
    else:
        return n


def poly_integral(poly, C=0):
    """ Takes an input list of polynomials, and calculates an output list
    of integrated polynomials.
    Example: x^3 + 3x +5 would be input as [5, 3, 0, 1]
    Return: [0, 5, 1.5, 0, 0.25]
    C is an integer representing the integration constant"""

    if isinstance(poly, list) and len(poly) > 0 and isinstance(C, int):
        return_list = [C]
        if len(poly) == 1 and poly[0] == 0:
            return return_list
        for i in range(0, len(poly)):
            return_list.append(format_number(poly[i] / (i + 1)))
        return return_list

    else:
        return None
