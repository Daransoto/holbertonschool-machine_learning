#!/usr/bin/env python3
""" This module contains the function create_layer. """
import tensorflow as tf


def create_layer(prev, n, activation):
    """ Creates a layer of a neural network. """
    he = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation, name='layer',
                            kernel_initializer=he)
    return layer(prev)
