#!/usr/bin/env python3
"""This module contains the function create_confusion_matrix."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix"""
    return np.dot(labels.T, logits)
