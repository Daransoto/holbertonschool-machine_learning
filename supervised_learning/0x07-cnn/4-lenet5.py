#!/usr/bin/env python3
""" This module contains the function lenet5. """
import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow.
    x is a tf.placeholder of shape (m, 28, 28, 1) containing the input images
     for the network.
        m is the number of images.
    y is a tf.placeholder of shape (m, 10) containing the one-hot labels for
     the network.
    The model consists of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding.
        Max pooling layer with kernels of shape 2x2 with 2x2 strides.
        Convolutional layer with 16 kernels of shape 5x5 with valid padding.
        Max pooling layer with kernels of shape 2x2 with 2x2 strides.
        Fully connected layer with 120 nodes.
        Fully connected layer with 84 nodes.
        Fully connected softmax output layer with 10 nodes.
    All layers requiring initialization initialize their kernels with the
     he_normal initialization method.
    All hidden layers requiring activation use the relu activation function.
    Returns:
        a tensor for the softmax activated output.
        a training operation that utilizes Adam optimization
         (with default hyperparameters).
        a tensor for the loss of the netowrk.
        a tensor for the accuracy of the network.
    """
    He = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                             activation=tf.nn.relu, kernel_initializer=He)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=He)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    pool2 = tf.layers.Flatten()(pool2)
    dense1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                             kernel_initializer=He)(pool2)
    dense2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                             kernel_initializer=He)(dense1)
    dense3 = tf.layers.Dense(units=10, kernel_initializer=He)(dense2)
    y_pred = tf.nn.softmax(dense3)
    y_pred_tag = tf.argmax(dense3, 1)
    y_tag = tf.argmax(y, 1)
    comp = tf.equal(y_pred_tag, y_tag)
    acc = tf.reduce_mean(tf.cast(comp, tf.float32))
    loss = tf.losses.softmax_cross_entropy(y, dense3)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    return y_pred, train_op, loss, acc
