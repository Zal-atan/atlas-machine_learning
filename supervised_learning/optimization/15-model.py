#!/usr/bin/env python3
""" This module creates model function and all needed assitant functions
"""
import tensorflow.compat.v1 as tf
import numpy as np


def create_layer(prev, n, activation):
    """
    Creates a layer for the neural network

    Inputs:
    prev - tensor output of the previous layer
    n - number of nodes in the layer to create
    activation - activation function that the layer should use

    Return:
    tensor output of the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    return tf.layers.dense(prev, n, activation=activation,
                           kernel_initializer=initializer)


def create_batch_norm_layer(prev, n, activation, epsilon, final):
    """
    Creates a batch normalization layer for a neural network in tensorflow

    Inputs:
    prev - activated output of the previous layer
    n - number of nodes in the layer to be created
    activation - activation function that should be used on the output
                 of the layer

    Returns:
    Tensor of the activated output for the layer
    """

    if activation is None:
        A = create_layer(prev, n, activation)
        return A

    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense_layer = tf.keras.layers.Dense(units=n, activation=None,
                                        kernel_initializer=init)

    z = dense_layer(prev)

    if final:
        return z

    mean, variance = tf.nn.moments(z, 0)
    gamma = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.zeros(n), trainable=True)

    normalize = tf.nn.batch_normalization(z, mean, variance,
                                          beta, gamma, epsilon)

    return activation(normalize)


def forward_prop(prev, layers, activations, epsilon):
    """
    Forward propogation function to be fed into model

    Inputs:
    input - input data
    layers - list containing number of nodes for each layer of neural network
    activations - list containing activation functions for each layer
    epsilon - very small number / avoids division by 0

    Returns:
    Tensor form of prediction of neural network
    """
    predict = prev

    for node in range(len(layers)):
        final = True if node == len(layers) - 1 else False

        predict = create_batch_norm_layer(predict,
                                          layers[node],
                                          activations[node],
                                          epsilon,
                                          final)

    return predict


def loss_calc(y, y_predict):
    """
    Function for calculating the loss of a prediction

    Inputs:
    y - placeholder for the labels of the input data
    y_predict - tensor format predictions of the network

    Return:
    Tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_predict)


def accuracy_calc(y, y_predict):
    """
    Calculates the accuracy of a prediction

    Inputs:
    y - placeholder for the labels of the input data
    y_predict - tensor format predictions of the network

    Return:
    Tensor containing the accuracy of the prediction
    """
    y = tf.math.argmax(y, axis=1)
    y_predict = tf.math.argmax(y_predict, axis=1)

    prediction = tf.equal(y, y_predict)
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return accuracy


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy

    Inputs:
    alpha - original learning rate
    decay_rate - weight used to determine the rate at which alpha will decay
    global_step - number of passes of gradient descent that have elapsed
    decay_step - number of passes of gradient descent that should
                 occur before alpha is decayed further

    Returns:
    The learning rate decay operation
    """
    learn_rate = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                             decay_rate, staircase=True)

    return learn_rate


def create_Adam_op(loss, alpha, beta1, beta2, epsilon, global_step):
    """
    Creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm:

    Inputs:
    loss - loss of the network
    alpha - learning rate
    beta1 - weight used for the first moment
    beta2 - weight used for the second moment
    epsilon - small number to avoid division by zero

    Returns:
    The Adam optimization operationy
    """
    optimized_Adam = tf.train.AdamOptimizer(alpha, beta1, beta2,
                                            epsilon).minimize(loss,
                                                              global_step)

    return optimized_Adam


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices in the same way

    Inputs:
    X - numpy.ndarray of shape (m, nx) to normalize
        m - the number of data points
        nx - number of features in X
    Y - numpy.ndarray of shape (m, ny) to normalize
        m - the same number of data points as in X
        ny - number of features in Y

    Returns:
    Shuffled X and Y matrices
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)

    return X[shuffle], Y[shuffle]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization

    Inputs:
    Data_train - tuple containing the training inputs and training labels,
                 respectively
    Data_valid - tuple containing the validation inputs and validation labels,
                 respectively
    layers - list containing the number of nodes in each layer of the network
    activation - list containing the activation functions used for each layer
                 of the network
    alpha - learning rate
    beta1 - weight for the first moment of Adam Optimization
    beta2 - weight for the second moment of Adam Optimization
    epsilon - small number used to avoid division by zero
    decay_rate - decay rate for inverse time decay of the learning rate
                 (the corresponding decay step should be 1)
    batch_size - number of data points that should be in a mini-batch
    epochs - number of times the training should pass through the whole dataset
    save_path - path where the model should be saved to

    Return:
    The path where the model was saved
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='x')
    tf.add_to_collection('x', x)
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name='y')
    tf.add_to_collection('y', y)

    y_predict = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_predict', y_predict)

    loss = loss_calc(y, y_predict)
    tf.add_to_collection('loss', loss)

    accuracy = accuracy_calc(y, y_predict)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon, global_step)
    tf.add_to_collection('train_op', train_op)

    mini_batch_size = int(np.ceil(X_train.shape[0] / batch_size))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs + 1):
            train_data = {x: X_train, y: Y_train}
            valid_data = {x: X_valid, y: Y_valid}

            train_cost = sess.run(loss, feed_dict=train_data)
            train_accuracy = sess.run(accuracy, feed_dict=train_data)
            valid_cost = sess.run(loss, feed_dict=valid_data)
            valid_accuracy = sess.run(accuracy, feed_dict=valid_data)

            print(f"After {i} epochs:")
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_accuracy}")
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy:  {valid_accuracy}")

            if i < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                for batch in range(mini_batch_size):
                    # update learning rate
                    sess.run(global_step.assign(i))
                    sess.run(alpha)

                    start = batch * batch_size
                    end = (batch + 1) * batch_size
                    if end > X_train.shape[0]:
                        end = X_train.shape[0]
                    X = X_shuffled[start:end]
                    Y = Y_shuffled[start:end]

                    # execute training for step
                    train_dict = {x: X, y: Y}
                    sess.run(train_op, feed_dict=train_dict)

                    if batch != 0 and (batch + 1) % 100 == 0:

                        batch_cost = sess.run(loss, feed_dict=train_dict)
                        batch_accuracy = sess.run(accuracy,
                                                  feed_dict=train_dict)

                        print(f"\tStep {batch + 1}:")
                        print(f"\t\tCost: {batch_cost}")
                        print(f"\t\tAccuracy: {batch_accuracy}")

        saver = tf.train.Saver()
        return saver.save(sess, save_path)
