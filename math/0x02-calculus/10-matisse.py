#!/usr/bin/env python3
""" This module contains the function poly_derivative. """


def poly_derivative(poly):
    """ Calculates the derivative of a polynomial.
    poly is a list of coefficients representing a polynomial.
    The index of the list represents the power of x that the
        coefficient belongs to, starting with x^0.
    Returns a new list of coefficients representing the derivative
        of the polynomial or None if not valid.
    """
    if type(poly) != list or poly == [] or type(poly[0]) not in [int, float]:
        return None
    if len(poly) == 1:
        return [0]
    res = []
    zeroes = True
    for idx, coef in enumerate(poly[1:]):
        if type(coef) not in [int, float]:
            return None
        res.append((idx + 1) * coef)
        if zeroes and res[-1]:
            zeroes = False
    if zeroes:
        return [0]
    return res
