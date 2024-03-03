#!/usr/bin/env python3
"""This module creates the lenet5 function"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow

    Inputs:
    * x is a tf.placeholder of shape (m, 28, 28, 1)
        containing the input images for the network
        * m is the number of images
    * y is a tf.placeholder of shape (m, 10) containing the one-hot

    Returns:
    Output of the pooling layer
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv_layer1 = tf.layers.conv2d(inputs=x,
                                   filters=6,
                                   kernel_size=(5, 5),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer=init)

    max_pool1 = tf.layers.max_pooling2d(inputs=conv_layer1,
                                        pool_size=(2, 2),
                                        strides=(2, 2))

    conv_layer2 = tf.layers.conv2d(inputs=max_pool1,
                                   filters=16,
                                   kernel_size=(5, 5),
                                   padding='valid',
                                   activation='relu',
                                   kernel_initializer=init)

    max_pool2 = tf.layers.max_pooling2d(inputs=conv_layer2,
                                        pool_size=(2, 2),
                                        strides=(2, 2))

    flat_pool = tf.layers.flatten(max_pool2)

    FC1 = tf.layers.dense(inputs=flat_pool,
                          units=120,
                          activation='relu',
                          kernel_initializer=init,
                          )

    FC2 = tf.layers.dense(inputs=FC1,
                          units=84,
                          activation='relu',
                          kernel_initializer=init,
                          )

    FC3 = tf.layers.dense(inputs=FC2,
                          units=10,
                          kernel_initializer=init)

    softmax = tf.nn.softmax(FC3)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y,
                                           logits=FC3)

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    y_max = tf.math.argmax(y, axis=1)
    y_pred_max = tf.math.argmax(FC3, axis=1)
    equality = tf.math.equal(y_max, y_pred_max)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    return softmax, optimizer, loss, accuracy
