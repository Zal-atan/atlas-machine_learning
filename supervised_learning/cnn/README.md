# This is a README for the Convolutional Neural Networks(CNN) repo.

### In this repo we will practicing uses of convolutional neural networks for machine Machine Learning
<br>

### Author - Ethan Zalta
<br>


# Tasks
### There are 6 tasks in this project

## Task 0
* Write a function def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)): that performs forward propagation over a convolutional layer of a neural network:

    * A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer
        * m is the number of examples
        * h_prev is the height of the previous layer
        * w_prev is the width of the previous layer
        * c_prev is the number of channels in the previous layer
    * W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels for the convolution
        * kh is the filter height
        * kw is the filter width
        * c_prev is the number of channels in the previous layer
        * c_new is the number of channels in the output
    * b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to the convolution
    * activation is an activation function applied to the convolution
    * padding is a string that is either same or valid, indicating the type of padding used
    * stride is a tuple of (sh, sw) containing the strides for the convolution
        * sh is the stride for the height
        * sw is the stride for the width
    * you may import numpy as np
    * Returns: the output of the convolutional layer

## Task 1
* Write a function def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'): that performs forward propagation over a pooling layer of a neural network:

    * A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer
        * m is the number of examples
        * h_prev is the height of the previous layer
        * w_prev is the width of the previous layer
        * c_prev is the number of channels in the previous layer
    * kernel_shape is a tuple of (kh, kw) containing the size of the kernel for the pooling
        * kh is the kernel height
        * kw is the kernel width
    * stride is a tuple of (sh, sw) containing the strides for the pooling
        * sh is the stride for the height
        * sw is the stride for the width
    * mode is a string containing either max or avg, indicating whether to perform maximum or     *   average pooling, respectively
    * you may import numpy as np
    * Returns: the output of the pooling layer

## Task 2
* Write a function def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)): that performs back propagation over a convolutional layer of a neural network:

    * dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial derivatives with respect to the unactivated output of the convolutional layer
        * m is the number of examples
        * h_new is the height of the output
        * w_new is the width of the output
        * c_new is the number of channels in the output
    * A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer
        * h_prev is the height of the previous layer
        * w_prev is the width of the previous layer
        * c_prev is the number of channels in the previous layer
    * W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels for the convolution
        * kh is the filter height
        * kw is the filter width
    * b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to the convolution
    * padding is a string that is either same or valid, indicating the type of padding used
stride is a tuple of (sh, sw) containing the strides for the convolution
        * sh is the stride for the height
        * sw is the stride for the width
    * you may import numpy as np
    * Returns: the partial derivatives with respect to the previous layer (dA_prev), the kernels (dW), and the biases (db), respectively

## Task 3
* Write a function def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'): that performs back propagation over a pooling layer of a neural network:

    * dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial derivatives with respect to the output of the pooling layer
        * m is the number of examples
        * h_new is the height of the output
        * w_new is the width of the output
        * c is the number of channels
    * A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing the output of the previous layer
        * h_prev is the height of the previous layer
        * w_prev is the width of the previous layer
    * kernel_shape is a tuple of (kh, kw) containing the size of the kernel for the pooling
        * kh is the kernel height
        * kw is the kernel width
    * stride is a tuple of (sh, sw) containing the strides for the pooling
        * sh is the stride for the height
        * sw is the stride for the width
    * mode is a string containing either max or avg, indicating whether to perform maximum or average pooling, respectively
    * you may import numpy as np
    * Returns: the partial derivatives with respect to the previous layer (dA_prev)

## Task 4
* Write a function def lenet5(x, y): that builds a modified version of the LeNet-5 architecture using tensorflow:

    * x is a tf.placeholder of shape (m, 28, 28, 1) containing the input images for the network
        * m is the number of images
    * y is a tf.placeholder of shape (m, 10) containing the one-hot labels for the network
    * The model should consist of the following layers in order:
        * Convolutional layer with 6 kernels of shape 5x5 with same padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Convolutional layer with 16 kernels of shape 5x5 with valid padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Fully connected layer with 120 nodes
        * Fully connected layer with 84 nodes
        * Fully connected softmax output layer with 10 nodes
    * All layers requiring initialization should initialize their kernels with the he_normal initialization method: tf.keras.initializers.VarianceScaling(scale=2.0)
    * All hidden layers requiring activation should use the relu activation function
    * you may import tensorflow.compat.v1 as tf
    * you may NOT use tf.keras only for the he_normal method.
    * Returns:
        * a tensor for the softmax activated output
        * a training operation that utilizes Adam optimization (with default hyperparameters)
        * a tensor for the loss of the netowrk
        * a tensor for the accuracy of the network

## Task 5
* Write a function def lenet5(X): that builds a modified version of the LeNet-5 architecture using keras:

    * X is a K.Input of shape (m, 28, 28, 1) containing the input images for the network
        * m is the number of images
    * The model should consist of the following layers in order:
        * Convolutional layer with 6 kernels of shape 5x5 with same padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Convolutional layer with 16 kernels of shape 5x5 with valid padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Fully connected layer with 120 nodes
        * Fully connected layer with 84 nodes
        * Fully connected softmax output layer with 10 nodes
    * All layers requiring initialization should initialize their kernels with the he_normal initialization method
    * All hidden layers requiring activation should use the relu activation function
you may import tensorflow.keras as K
    * Returns: a K.Model compiled to use Adam optimization (with default hyperparameters) and accuracy metrics

## Task 6
* A common practice in the machine learning industry is to read and review journal articles on a weekly basis. Read and write a summary of Krizhevsky et. al.‘s 2012 paper ImageNet Classification with Deep Convolutional Neural Networks. Your summary should include:

    * Introduction: Give the necessary background to the study and state its purpose.
    * Procedures: Describe the specifics of what this study involved.
    * Results: In your own words, discuss the major findings and results.
    * Conclusion: In your own words, summarize the researchers’ conclusions.
    * Personal Notes: Give your reaction to the study.
Your posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on Twitter and LinkedIn.

When done, please add all URLs below (blog post, tweet, etc.)

Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.
