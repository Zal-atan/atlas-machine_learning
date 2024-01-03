#!/usr/bin/env python3
"""This module uses numpy to added matrices"""
import numpy as np


def add_matrices(mat1, mat2):
    try:
        return np.add(mat1, mat2)
    except Exception:
        return None
