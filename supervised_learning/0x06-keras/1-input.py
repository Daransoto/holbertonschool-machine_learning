#!/usr/bin/env python3
""" This module contains the function build_model. """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library without using Sequential.
    nx is the number of input features to the network.
    layers is a list containing the number of nodes in each layer of the
     network.
    activations is a list containing the activation functions used for each
     layer of the network.
    lambtha is the L2 regularization parameter.
    keep_prob is the probability that a node will be kept for dropout.
    Returns: the keras model.
    """
    _input = K.Input(shape=(nx,))
    regs = K.regularizers.L1L2(l2=lambtha)
    layer = K.layers.Dense(units=layers[0], activation=activations[0],
                           kernel_regularizer=regs, input_shape=(nx,))(_input)
    for i in range(1, len(layers)):
        layer = K.layers.Dropout(1 - keep_prob)(layer)
        layer = K.layers.Dense(units=layers[i], activation=activations[i],
                               kernel_regularizer=regs)(layer)
    NN = K.Model(inputs=_input, outputs=layer)
    return NN
