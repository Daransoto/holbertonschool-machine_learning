#!/usr/bin/env python3
""" This module contains the function model. """
import tensorflow as tf
import numpy as np


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using Adam
     optimization, mini-batch gradient descent, learning rate decay, and batch
     normalization.
    Data_train is a tuple containing the training inputs and training labels,
     respectively.
    Data_valid is a tuple containing the validation inputs and validation
     labels, respectively.
    layers is a list containing the number of nodes in each layer of the
     network.
    activation is a list containing the activation functions used for each
     layer of the network.
    alpha is the learning rate.
    beta1 is the weight for the first moment of Adam Optimization.
    beta2 is the weight for the second moment of Adam Optimization.
    epsilon is a small number used to avoid division by zero.
    decay_rate is the decay rate for inverse time decay of the learning rate
     (the corresponding decay step should be 1).
    batch_size is the number of data points that should be in a mini-batch.
    epochs is the number of times the training should pass through the whole
     dataset.
    save_path is the path where the model should be saved to.
    Returns: the path where the model was saved.
    """
    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layers, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)
    saver = tf.train.Saver()
    initializer = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initializer)
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
            if i < epochs:
                sess.run(global_step.assign(epoch))
                sess.run(alpha)
                newX, newY = shuffle_data(X_train, Y_train)
                m = newX.shape[0]
                start = 0
                step = 1
                while m > 0:
                    if m - batch_size < 0:
                        end = newX.shape[0]
                    else:
                        end = start + batch_size
                    sess.run(train_op, feed_dict={x: newX[start:end],
                                                  y: newY[start:end]})
                    if step % 100 == 0:
                        step_acc = sess.run(accuracy,
                                            feed_dict={x: newX[start:end],
                                                       y: newY[start:end]})
                        step_cost = sess.run(loss,
                                             feed_dict={x: newX[start:end],
                                                        y: newY[start:end]})
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_acc))
                    start += batch_size
                    step += 1
                    m -= batch_size
        return saver.save(sess, save_path)


def create_placeholders(nx, classes):
    """ Returns two placeholders, x and y, for a neural network. """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.
    prev is the activated output of the previous layer.
    n is the number of nodes in the layer to be created.
    activation is the activation function that should be used on the output of
     the layer.
    Returns: a tensor of the activated output for the layer.
    """
    if not activation:
        return create_layer(prev, n, activation)
    He = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=He, name='layer')
    epsilon = 1e-8
    net = layer(prev)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    mean, var = tf.nn.moments(net, axes=[0])
    z = tf.nn.batch_normalization(net, mean=mean, variance=var, offset=beta,
                                  scale=gamma, variance_epsilon=epsilon)
    return activation(z)


def forward_prop(x, layer_sizes=[], activations=[]):
    """ Creates the forward propagation graph for the neural network. """
    current = x
    for i, l_size in enumerate(layer_sizes):
        current = create_batch_norm_layer(current, l_size, activations[i])
    return current


def calculate_accuracy(y, y_pred):
    """ Calculates the accuracy of a prediction. """
    y_hot = tf.argmax(y, axis=1)
    y_p_hot = tf.argmax(y_pred, axis=1)
    comp = tf.cast(tf.equal(y_hot, y_p_hot), dtype=tf.float32)
    return tf.reduce_mean(comp)


def calculate_loss(y, y_pred):
    """ Calculates the softmax cross-entropy loss of a prediction. """
    return tf.losses.softmax_cross_entropy(y, logits=y_pred)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using the
     Adam optimization algorithm.
    loss is the loss of the network.
    alpha is the learning rate.
    beta1 is the weight used for the first moment.
    beta2 is the weight used for the second moment.
    epsilon is a small number to avoid division by zero.
    Returns: the Adam optimization operation.
    """
    return tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                  beta2=beta2, epsilon=epsilon).minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation in tensorflow using inverse time
     decay in a stepwise fashion.
    alpha is the original learning rate.
    decay_rate is the weight used to determine the rate at which alpha will
     decay.
    global_step is the number of passes of gradient descent that have elapsed.
    decay_step is the number of passes of gradient descent that should occur
     before alpha is decayed further.
    Returns: the learning rate decay operation.
    """
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate, staircase=True)


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.
    X is the first numpy.ndarray of shape (m, nx) to shuffle.
        m is the number of data points.
        nx is the number of features in X.
    Y is the second numpy.ndarray of shape (m, ny) to shuffle.
        m is the same number of data points as in X.
        ny is the number of features in Y.
    Returns: the shuffled X and Y matrices.
    """
    indices = np.random.permutation(X.shape[0])
    return X[indices], Y[indices]


def create_layer(prev, n, activation):
    """ Creates a layer of a neural network. """
    he = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation, name='layer',
                            kernel_initializer=he)
    return layer(prev)
