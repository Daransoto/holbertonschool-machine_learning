#!/usr/bin/env python3
""" This module contains the function forward_prop. """
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """ Creates the forward propagation graph for the neural network. """
    create_layer = __import__('1-create_layer').create_layer
    current = x
    for i, l_size in enumerate(layer_sizes):
        current = create_layer(current, l_size, activations[i])
    return current
