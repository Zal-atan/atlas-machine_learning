#!/usr/bin/env python3
""" This module creates train_mini_batch(X_train, Y_train, X_valid, Y_valid,
batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
save_path="/tmp/model.ckpt"): function"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent

    Inputs:
    X_train - numpy.ndarray of shape (m, 784) containing the training data
        m - number of data points
        784 - number of input features
    Y_train - one-hot numpy.ndarray of shape (m, 10) containing training labels
        10 - number of classes the model should classify
    X_valid - numpy.ndarray of shape (m, 784) containing the validation data
    Y_valid - one-hot numpy.ndarray of shape (m, 10) containing
              the validation labels
    batch_size - number of data points in a batch
    epochs - number of times the training should pass through the whole dataset
    load_path - path from which to load the model
    save_path - path to where the model should be saved after training

    Returns:
    The path where the model was saved
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)  # Reload wights from save

        # Reload attributes from save
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        # Get even batch size
        mini_batch_size = len(X_train)//batch_size
        while mini_batch_size % batch_size != 0:
            mini_batch_size += 1

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
            print(f"\tValidation Accuracy: {valid_accuracy}")

            if i < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                for batch in range(mini_batch_size):
                    start = batch_size * batch
                    end = batch_size * (batch + 1)
                    train_batch = X_shuffled[start:end]
                    train_label = Y_shuffled[start:end]
                    train_dict = {x: train_batch, y: train_label}

                    batch_train = sess.run(train_op, train_dict)

                    if batch % 100 == 0:
                        if batch == 0:
                            continue
                        batch_cost = sess.run(loss, train_dict)
                        batch_accuracy = sess.run(accuracy, train_dict)

                        print(f"\tStep {batch}:")
                        print(f"\t\tCost: {batch_cost}")
                        print(f"\t\tAccuracy: {batch_accuracy}")

        save_path = saver.save(sess, save_path)
        return save_path
