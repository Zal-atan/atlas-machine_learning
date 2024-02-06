#!/usr/bin/env python3
"""
Create a function train(X_train, Y_train, X_valid, Y_valid, layer_sizes, 
activations, alpha, iterations, save_path="/tmp/model.ckpt"):
"""
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, Trains, and Saves a neural network classifier

    Inputs:
    X_train - numpy.ndarray containing the training input data
    Y_train - numpy.ndarray containing the training labels
    X_valid - numpy.ndarray containing the validation input data
    Y_valid - numpy.ndarray containing the validation labels
    layer_sizes - list containing number of nodes in each layer of the network
    activations - list containing activation funcs for each layer of network
    alpha - learning rate
    iterations - number of iterations to train over
    save_path - designates where to save the model

    Return:
    path where the model was returned
    """

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    for iter in range(iterations + 1):
        train_loss = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        valid_loss = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

        if iter == iterations or iter % 100 == 0:
            print(f"After {iter} iterations:")
            print(f"\tTraining Cost: {train_loss}")
            print(f"\tTraining Accuracy: {train_accuracy}")
            print(f"\tValidation Cost: {valid_loss}")
            print(f"\tValidation Accuracy: {valid_accuracy}")
                                                        
        if iter < iterations:
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

    return saver.save(sess, save_path)

