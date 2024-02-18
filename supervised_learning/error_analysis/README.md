# This is a README for the Error Analysis repo.

### In this repo we will practicing basic uses of Confusion Matrices and handling errors in Machine Learning
<br>

### Author - Ethan Zalta
<br>


# Tasks
### There are 7 tasks in this project

## Task 0
* Write the function def create_confusion_matrix(labels, logits): that creates a confusion matrix:

    * labels is a one-hot numpy.ndarray of shape (m, classes) containing the correct labels for each data point
        * m is the number of data points
        * classes is the number of classes
    * logits is a one-hot numpy.ndarray of shape (m, classes) containing the predicted labels
    * Returns: a confusion numpy.ndarray of shape (classes, classes) with row indices representing the correct labels and column indices representing the predicted labels

## Task 1
* Write the function def sensitivity(confusion): that calculates the sensitivity for each class in a confusion matrix:

    * confusion is a confusion numpy.ndarray of shape (classes, classes) where row indices represent the correct labels and column indices represent the predicted labels
        * classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the sensitivity of each class

## Task 2
* Write the function def precision(confusion): that calculates the precision for each class in a confusion matrix:

    * confusion is a confusion numpy.ndarray of shape (classes, classes) where row indices represent the correct labels and column indices represent the predicted labels
        * classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the precision of each class

## Task 3
* Write the function def specificity(confusion): that calculates the specificity for each class in a confusion matrix:

    * confusion is a confusion numpy.ndarray of shape (classes, classes) where row indices represent the correct labels and column indices represent the predicted labels
        * classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the specificity of each class

## Task 4
* Write the function def f1_score(confusion): that calculates the F1 score of a confusion matrix:

    * confusion is a confusion numpy.ndarray of shape (classes, classes) where row indices represent the correct labels and column indices represent the predicted labels
        * classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the F1 score of each class
    * You must use sensitivity = __import__('1-sensitivity').sensitivity and precision = __import__('2-precision').precision create previously

## Task 5
* In the text file 5-error_handling, write the lettered answer to the question of how you should approach the following scenarios. Please write the answer to each scenario on a different line. If there is more than one way to approach a scenario, please use CSV formatting and place your answers in alphabetical order (ex. A,B,C):

## Task 6
* Given the following training and validation confusion matrices and the fact that human level performance has an error of ~14%, determine what the most important issue is and write the lettered answer in the file 6-compare_and_contrast
