# This is a README for the calculus repo.

### In this repo we will practicing calculus as well as applications in Python.
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
### There are 18 tasks in this project

## Task 0
* $\sum_{i=2}^{5} i$

    1. 3 + 4 + 5
    1. 3 + 4
    1. 2 + 3 + 4 + 5
    1. 2 + 3 + 4

## Task 1
* $\sum_{k=1}^{4} 9i - 2k$

    1. 90 - 20
    1. 36i - 20
    1. 90 - 8k
    1. 36i - 8k

## Task 2
* $\prod_{i = 1}^{m} i$

    1. (m - 1)!
    1. 0
    1. (m + 1)!
    1. m!

## Task 3
* $\prod_{i = 0}^{10} i$

    1. 10!
    1. 9!
    1. 100
    1. 0

## Task 4
* $\frac{dy}{dx} where y = x^4 + 3x^3 - 5x + 1$

    1. 3x^3 + 6x^2 -4
    1. 4x^3 + 6x^2 - 5
    1. 4x^3 + 9x^2 - 5
    1. 4x^3 + 9x^2 - 4

## Task 5
* $\frac{d (xln(x))}{dx}$

    1. ln(x)
    1. \frac{1}{x} + 1
    1. ln(x) + 1
    1. \frac{1}{x}

## Task 6
* $\frac{d (ln(x^2))}{dx}$

    1. \frac{2}{x}
    1. \frac{1}{x^2}
    1. \frac{2}{x^2}
    1. \frac{1}{x}

## Task 7
* $\frac{\partial f(x, y)}{\partial y}$ where f(x, y) = e^{xy} and $\frac{\partial x}{\partial y}$=$\frac{\partial y}{\partial x}=0$

    1. e^{xy}
    1. ye^{xy}
    1. xe^{xy}
    1. e^{x}

## Task 8
* $\frac{\partial^2}{\partial y\partial x}(e^{x^2y})$ where $\frac{\partial x}{\partial y}=\frac{\partial y}{\partial x}=0$

    1. 2x(1+y)e^{x^2y}
    1. 2xe^{2x}
    1. 2x(1+x^2y)e^{x^2y}
    1. e^{2x}

## Task 9
* Write a function def summation_i_squared(n): that calculates $\sum_{i=1}^{n} i^2$:

    1. n is the stopping condition
    1. Return the integer value of the sum
    1. If n is not a valid number, return None
    1. You are not allowed to use any loops

## Task 10
* Write a function def poly_derivative(poly): that calculates the derivative of a polynomial:

    * poly is a list of coefficients representing a polynomial
        * the index of the list represents the power of x that the coefficient belongs to
        * Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
    * If poly is not valid, return None
    * If the derivative is 0, return [0]
    * Return a new list of coefficients representing the derivative of the polynomial

## Task 11
* $\int x^3 \, dx$

    * 3x2 + C
    * x4/4 + C
    * x4 + C
    * x4/3 + C

## Task 12
* $\int e^{2y} \, dy$
    * e2y + C
    * ey + C
    * e2y/2 + C
    * ey/2 + C

## Task 13
* $\int_{0}^{3} u^2 \, du$
    * 3
    * 6
    * 9
    * 27

## Task 14
* $\int_{-1}^{0} \frac{1}{v} \, dv$
    * -1
    * 0
    * 1
    * undefined

## Task 15
* $\int_{0}^{5} x\, dy$

    * 5
    * 5x
    * 25
    * 25x

## Task 16
* $\int_{1}^{2} \int_{0}^{3} x^2 y^{-1} \, dx \, dy$

    * 9ln(2)
    * 9
    * 27ln(2)
    * 27

## Task 17
* Write a function def poly_integral(poly, C=0): that calculates the integral of a polynomial:

    * poly is a list of coefficients representing a polynomial
        * the index of the list represents the power of x that the coefficient belongs to
        * Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
    * C is an integer representing the integration constant
    * If a coefficient is a whole number, it should be represented as an integer
    * If poly or C are not valid, return None
    * Return a new list of coefficients representing the integral of the polynomial
    * The returned list should be as small as possible        *
