#!/usr/bin/env python3
""" This module contains the function calculate_accuracy. """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ Calculates the accuracy of a prediction. """
    y_hot = tf.argmax(y, axis=1)
    y_p_hot = tf.argmax(y_pred, axis=1)
    comp = tf.cast(tf.equal(y_hot, y_p_hot), dtype=tf.float32)
    return tf.reduce_mean(comp)
