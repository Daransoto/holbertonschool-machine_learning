#!/usr/bin/env python3
""" This module contains the function train_model. """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.
    network is the model to train.
    data is a numpy.ndarray of shape (m, nx) containing the input data.
    labels is a one-hot numpy.ndarray of shape (m, classes) containing the
     labels of data.
    batch_size is the size of the batch used for mini-batch gradient descent.
    epochs is the number of passes through data for mini-batch gradient
     descent.
    validation_data is the data to validate the model with, if not None.
    early_stopping is a boolean that indicates whether early stopping should be
     used, only if validation_data exists, and based on validation loss.
    patience is the patience used for early stopping.
    learning_rate_decay is a boolean that indicates whether learning rate
     decay should be used (if vatlidation_data exists and with the inverse time
     decay in a stepwise fashion, printing a message on each update).
    alpha is the initial learning rate.
    decay_rate is the decay rate.
    verbose is a boolean that determines if output should be printed during
     training.
    shuffle is a boolean that determines whether to shuffle the batches every
     epoch.
    Returns: the History object generated after training the model.
    """
    def lrd(epoch):
        """ Function for learning rate decay. """
        return alpha / (1 + decay_rate * epoch)

    callbacks = []

    if validation_data:

        if early_stopping:

            callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                       mode='min',
                                                       patience=patience))

        if learning_rate_decay:

            callbacks.append(K.callbacks.LearningRateScheduler(lrd, verbose=1))

    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callbacks)
