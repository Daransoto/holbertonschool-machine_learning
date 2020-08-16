#!/usr/bin/env python3
""" This module contains the function calculate_loss. """
import tensorflow as tf


def create_train_op(loss, alpha):
    """ Creates the training operation for the network. """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
