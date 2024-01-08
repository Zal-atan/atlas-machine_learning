#!/usr/bin/env python3
"""Creating a functions which calculates sum of the total of the square of
1 to n"""


def summation_i_squared(n):
    """Takes the squares of all values of from 1 to n and returns a sum of
       each"""
    if not isinstance(n, int) or (n < 1):
        return None
    if n == 1:
        return n
    return n**2 + summation_i_squared(n-1)
