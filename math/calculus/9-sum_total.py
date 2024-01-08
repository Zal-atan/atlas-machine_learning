#!/usr/bin/env python3
""" In this module I will create a summation of i^2 function"""


def summation_i_squared(n):
    """Takes the squares of all values of from 1 to n and returns a sum of
       each"""
    if not isinstance(n, int):
        return None
    if n == 1:
        return n
    return n**2 + summation_i_squared(n-1)
