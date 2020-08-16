#!/usr/bin/env python3
""" This module contains the function calculate_loss. """
import tensorflow as tf


def calculate_loss(y, y_pred):
    """ Calculates the softmax cross-entropy loss of a prediction. """
    return tf.losses.softmax_cross_entropy(y, logits=y_pred)
