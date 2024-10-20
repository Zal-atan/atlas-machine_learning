#!/usr/bin/env python3
""" This modules creates an add_arrays function"""


def add_arrays(arr1, arr2):
    """Add two arrays element-wise, returns None if arrays are not the same
    shape. Takes two arrays as arguments"""
    if len(arr1) != len(arr2):
        return None
    sum_array = [arr1[i] + arr2[i] for i in range(len(arr1))]
    return sum_array
