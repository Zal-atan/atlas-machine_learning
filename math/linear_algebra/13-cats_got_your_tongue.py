#!/usr/bin/env python3
""" This modules uses numpy to concatenate two strings"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Takes two input matrices and an optional input axis to concanate
    a matrix which will be returned"""
    return np.concatenate((mat1, mat2), axis)
