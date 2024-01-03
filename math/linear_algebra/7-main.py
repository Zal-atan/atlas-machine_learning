#!/usr/bin/env python3
import sys
cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D
m1 = [[4, -7, 56, 2], [5, 106, 7, 2]]
m2 = [[2, -6, 3], [0, -6, 3]]
m = cat_matrices2D(m1, m2)
print(m)
m1 = [[484, 247], [554, 16], [5, 88]]
m2 = [[233, -644, 325], [406, -16, 33], [765, 34, -39]]
m = cat_matrices2D(m1, m2, axis=0)
print(m)
m1 = [[-54, -87, 95], [54, 16, -72]]
m2 = [[12, 63, 79], [-10, 69, -9], [76, 45, -11]]
m = cat_matrices2D(m1, m2, axis=1)
print(m)
