#!/usr/bin/env python3
"""This module contains the function specificity"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.
    confusion is a confusion numpy.ndarray of shape (classes, classes) where
     row indices represent the correct labels and column indices represent the
     predicted labels.
        classes is the number of classes.
    Returns: a numpy.ndarray of shape (classes,) containing the specificity of}
     each class
    """
    true_pos = np.diag(confusion)
    false_neg = np.sum(confusion, axis=1) - true_pos
    false_pos = np.sum(confusion, axis=0) - true_pos
    true_neg = np.sum(confusion) - (false_pos + false_neg + true_pos)
    SPECIFICITY = TN / (FP + TN)

    return true_neg / (false_pos + true_neg)
