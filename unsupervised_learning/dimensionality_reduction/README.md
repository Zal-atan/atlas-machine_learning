# This is a README for the Dimensionality Reduction repo.

### In this repo we will practicing uses of reducing dimensionality for interpretability for machine Machine Learning
<br>

### Author - Ethan Zalta
<br>


# Tasks
### There are 2 tasks in this project

## Task 0
* Write a function def pca(X, var=0.95): that performs PCA on a dataset:

    * X is a numpy.ndarray of shape (n, d) where:
        * n is the number of data points
        * d is the number of dimensions in each point
        * all dimensions have a mean of 0 across all data points
    * var is the fraction of the variance that the PCA transformation should maintain
    * Returns: the weights matrix, W, that maintains var fraction of X‘s original variance
    * W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality of the transformed X

## Task 1
* Write a function def pca(X, ndim): that performs PCA on a dataset:

    * X is a numpy.ndarray of shape (n, d) where:
        * n is the number of data points
        * d is the number of dimensions in each point
    * ndim is the new dimensionality of the transformed X
    * Returns: T, a numpy.ndarray of shape (n, ndim) containing the transformed version of X

