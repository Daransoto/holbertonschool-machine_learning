#!/usr/bin/env python3
""" This module contains the functions save_config and load_config. """
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a models configuration in JSON format.
    network is the model whose configuration should be saved.
    filename is the path of the file that the configuration should be saved to.
    Returns: None.
    """
    with open(filename, 'w') as f:
        f.write(network.to_json())
    return None


def load_weights(network, filename):
    """
    Loads a model with a specific configuration.
    filename is the path of the file containing the models configuration in
     JSON format.
    Returns: the loaded model.
    """
    with open(filename, "r") as f:
        return K.models.model_from_json(f.read())
