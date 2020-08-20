#!/usr/bin/env python3
""" This module contains the function train_mini_batch. """
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent.
    X_train is a numpy.ndarray of shape (m, 784) containing the training data.
        m is the number of data points.
        784 is the number of input features.
    Y_train is a one-hot numpy.ndarray of shape (m, 10) containing the training
     labels.
        10 is the number of classes the model should classify.
    X_valid is a numpy.ndarray of shape (m, 784) containing the validation
     data.
    Y_valid is a one-hot numpy.ndarray of shape (m, 10) containing the
     validation labels.
    batch_size is the number of data points in a batch.
    epochs is the number of times the training should pass through the whole
     dataset.
    load_path is the path from which to load the model.
    save_path is the path to where the model should be saved after training.
    Returns: the path where the model was saved
    """
    m = X_train.shape[0]
    saved = tf.train.import_meta_graph("{}.meta".format(load_path))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saved.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        for i in range(epochs + 1):
            train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            train_loss = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            val_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            val_loss = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_loss))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(val_loss))
            print("\tValidation Accuracy: {}".format(val_acc))
            if i != epochs:
                newX, newY = shuffle_data(X_train, Y_train)
                start = 0
                end = batch_size
                step = 0
                while (end <= m):
                    sess.run(train_op, feed_dict={x: newX[start:end],
                                                  y: newY[start:end]})
                    if (step + 1) % 100 == 0:
                        step_acc = sess.run(accuracy,
                                            feed_dict={x: newX[start:end],
                                                       y: newY[start:end]})
                        step_cost = sess.run(loss,
                                             feed_dict={x: newX[start:end],
                                                        y: newY[start:end]})
                        print("\tStep {}:".format(step + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_acc))
                    start = end
                    if (end + batch_size <= m):
                        end += batch_size
                    else:
                        end += m % batch_size
                    step += 1

        return saver.save(sess, save_path)
