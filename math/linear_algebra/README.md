# This is a README for the linear algebra repo.

### In this repo we will practicing linear algebra as well as applications in Python.
<br>

### Author - Ethan Zalta
<br>

# Docker Download Instruction

### To download this repo in a useable form with all requirements satisfied:
* docker pull zalatan/linear_algebra:0.1

### To run this program in interactive mode to start the files manually
* docker run -it zalatan/linear_algebra:0.1

### To run individual files:
```
python3 2-main.py
# or
./2-main.py
```
<br>

# Tasks
### There are 15 tasks in this project with 3 Bonus Tasks

## Task 0
* Complete the following source code (found below):

    * arr1 should be the first two numbers of arr
    * arr2 should be the last five numbers of arr
    * arr3 should be the 2nd through 6th numbers of arr
    * You are not allowed to use any loops or conditional statements
    * Your program should be exactly 8 lines

## Task 1
* Complete the following source code (found below):

    * the_middle should be a 2D matrix containing the 3rd and 4th columns of matrix
    * You are not allowed to use any conditional statements
    * You are only allowed to use one for loop
    * Your program should be exactly 6 lines

## Task 2
* Write a function def matrix_shape(matrix): that calculates the shape of a matrix:

    * You can assume all elements in the same dimension are of the same type/shape
    * The shape should be returned as a list of integers

## Task 3
* Write a function def matrix_transpose(matrix): that returns the transpose of a 2D matrix, matrix:

    * You must return a new matrix
    * You can assume that matrix is never empty
    * You can assume all elements in the same dimension are of the same type/shape

## Task 4
* Write a function def add_arrays(arr1, arr2): that adds two arrays element-wise:

    * You can assume that arr1 and arr2 are lists of ints/floats
    * You must return a new list
    * If arr1 and arr2 are not the same shape, return None

## Task 5
* Write a function def add_matrices2D(mat1, mat2): that adds two matrices element-wise:

    * You can assume that mat1 and mat2 are 2D matrices containing ints/floats
    * You can assume all elements in the same dimension are of the same type/shape
    * You must return a new matrix
    * If mat1 and mat2 are not the same shape, return None

## Task 6
* Write a function def cat_arrays(arr1, arr2): that concatenates two arrays:

    * You can assume that arr1 and arr2 are lists of ints/floats
    * You must return a new list

## Task 7
* Write a function def cat_matrices2D(mat1, mat2, axis=0): that concatenates two matrices along a specific axis:

    * You can assume that mat1 and mat2 are 2D matrices containing ints/floats
    * You can assume all elements in the same dimension are of the same type/shape
    * You must return a new matrix
    * If the two matrices cannot be concatenated, return None

## Task 8
* Write a function def mat_mul(mat1, mat2): that performs matrix multiplication:

    * You can assume that mat1 and mat2 are 2D matrices containing ints/floats
    * You can assume all elements in the same dimension are of the same type/shape
    * You must return a new matrix
    * If the two matrices cannot be multiplied, return None

## Task 9
* Complete the following source code (found below):

    * mat1 should be the middle two rows of matrix
    * mat2 should be the middle two columns of matrix
    * mat3 should be the bottom-right, square, 3x3 matrix of matrix
    * You are not allowed to use any loops or conditional statements
    * Your program should be exactly 10 lines

## Task 10
* Write a function def np_shape(matrix): that calculates the shape of a numpy.ndarray:

    * You are not allowed to use any loops or conditional statements
    * You are not allowed to use try/except statements
    * The shape should be returned as a tuple of integers

## Task 11
* Write a function def np_transpose(matrix): that transposes matrix:

    * You can assume that matrix can be interpreted as a numpy.ndarray
    * You are not allowed to use any loops or conditional statements
    * You must return a new numpy.ndarray

## Task 12
* Write a function def np_elementwise(mat1, mat2): that performs element-wise addition, subtraction, multiplication, and division:

    * You can assume that mat1 and mat2 can be interpreted as numpy.ndarrays
    * You should return a tuple containing the element-wise sum, difference, product, and quotient, respectively
    * You are not allowed to use any loops or conditional statements
    * You can assume that mat1 and mat2 are never empty

## Task 13
* Write a function def np_cat(mat1, mat2, axis=0) that concatenates two matrices along a specific axis:

    * You can assume that mat1 and mat2 can be interpreted as numpy.ndarrays
    * You must return a new numpy.ndarray
    * You are not allowed to use any loops or conditional statements
    * You may use: import numpy as np
    * You can assume that mat1 and mat2 are never empty

## Task 14
* Write a function def np_matmul(mat1, mat2): that performs matrix multiplication:

    * You can assume that mat1 and mat2 are numpy.ndarrays
    * You are not allowed to use any loops or conditional statements
    * You may use: import numpy as np
    *   You can assume that mat1 and mat2 are never empty

## Bonus Task 100
* Write a function def np_slice(matrix, axes={}): that slices a matrix along specific axes:

    * You can assume that matrix is a numpy.ndarray
    * You must return a new numpy.ndarray
    * axes is a dictionary where the key is an axis to slice along and the value is a tuple representing the slice to make along that axis
    * You can assume that axes represents a valid slice

## Bonus Task 101
* Write a function def add_matrices(mat1, mat2): that adds two matrices:

    * You can assume that mat1 and mat2 are matrices containing ints/floats
    * You can assume all elements in the same dimension are of the same type/shape
    * You must return a new matrix
    * If matrices are not the same shape, return None
    * You can assume that mat1 and mat2 will never be empty

## Bonus Tasks 102
* Write a function def cat_matrices(mat1, mat2, axis=0): that concatenates two matrices along a specific axis:

    * You can assume that mat1 and mat2 are matrices containing ints/floats
    * You can assume all elements in the same dimension are of the same type/shape
    * You must return a new matrix
    * If you cannot concatenate the matrices, return None
    * You can assume that mat1 and mat2 are never empty
