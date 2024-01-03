#!/usr/bin/env python3
"""Module for creating an array slicing function"""
import numpy as np


def np_slice(matrix, axes={}):
    """Returns a sliced matrix, based on the input matrix, and split based
    on the dictionary input as axes"""
    copy_matrix = np.copy(matrix)

    for axi, slice_info in axes.items():
        # if axi is None:
        #     axi = 0

        result = np.take(copy_matrix, indices=slice_info, axis=axi)
        return result
