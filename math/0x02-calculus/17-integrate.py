#!/usr/bin/env python3
""" This module contains the function poly_integral. """


def poly_integral(poly, C=0):
    """ Calculates the integral of a polynomial.
    poly is a list of coefficients representing a polynomial.
    The index of the list represents the power of x that the
        coefficient belongs to, starting with x^0.
    C is an integer representing the integration constant.
    Returns a new list of coefficients representing the integral
        of the polynomial or None if not valid.
    """
    if type(poly) != list or poly == [] or type(C) not in [int, float]:
        return None
    res = []
    intC = int(C)
    if intC == C:
        res.append(intC)
    else:
        res.append(C)
    for idx, coef in enumerate(poly):
        if type(coef) not in [int, float]:
            return None
        new = coef / (idx + 1)
        intC = int(new)
        if intC == new:
            res.append(intC)
        else:
            res.append(new)
    while res and res[-1] == 0:
        res.pop()
    if not res:
        return [0]
    return res
