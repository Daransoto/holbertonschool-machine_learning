#!/usr/bin/env python3
""" This module contains the function evaluate. """
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ Evaluates the output of a neural network. """
    saved = tf.train.import_meta_graph("{}.meta".format(save_path))
    with tf.Session() as sess:
        saved.restore(sess, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        n_pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        n_acc = sess.run(accuracy, feed_dict={x: X, y: Y})
        n_loss = sess.run(loss, feed_dict={x: X, y: Y})
        return n_pred, n_acc, n_loss
