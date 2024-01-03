#!/usr/bin/env python3
import sys
cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D
m1 = [[1]]
m2 = [[2]]
m = cat_matrices2D(m1, m2)
if type(m) is not list or m is m1 or m is m2 or not len(m) or type(m[0]) is not list:
    print("Not a new matrix")
    sys.exit(1)
print(m)
m1 = [[4, -7, 56, 2], [5, 106, 7, 2]]
m2 = [[2, -6, 3, 23], [0, -6, 3, 42], [73, 8, 2, 99]]
m = cat_matrices2D(m1, m2)
if type(m) is not list or m is m1 or m is m2 or not len(m) or type(m[0]) is not list:
    print("Not a new matrix")
    sys.exit(1)
print(m)
m1 = [[484, 247, -556], [554, 16, 75], [5, 88, 23]]
m2 = [[233, -644, 325], [406, -16, 33]]
m = cat_matrices2D(m1, m2, axis=0)
if type(m) is not list or m is m1 or m is m2 or not len(m) or type(m[0]) is not list:
    print("Not a new matrix")
    sys.exit(1)
print(m)
m1 = [[-54, -87, 56, -92, 81], [54, 16, -72, 42, 901]]
m2 = [[12, 63], [-10, 69]]
m = cat_matrices2D(m1, m2, axis=1)
if type(m) is not list or m is m1 or m is m2 or not len(m) or type(m[0]) is not list:
    print("Not a new matrix")
    sys.exit(1)
print(m)
