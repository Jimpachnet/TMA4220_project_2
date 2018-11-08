#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements the P1 reference element
"""

import numpy as np


class P1ReferenceElement:
    """
    Class representing the P1 reference element
    """

    def value(self, x):
        """
        Returns the value of the three shape functions at x
        :param x: A tuple of the (x,y,z) coordinate
        :return: An array with four values of the shape functions
        """
        if (x[0] + x[1] +x[2] <= 1 and x[0] >= 0 and x[1] >= 0 and x[2] >= 0):
            p_0 = 1 - x[0] - x[1] - x[2]
            p_1 = x[0]
            p_2 = x[1]
            p_3 = x[2]
        else:
            p_0 = 0
            p_1 = 0
            p_2 = 0
            p_3 = 0
        return np.array([p_0, p_1, p_2, p_3])

    def gradients(self, x):
        """
        Calculates the gradient of the reference element
        :param x: The position at which the gradient should be calculated
        :return: Matrix 2x3 containing the gradients
        """
        if (x[0] + x[1]+x[2] <= 1 and x[0] >= 0 and x[1] >= 0 and x[2] >=0):
            grad = np.matrix('-1 1 0 0; -1 0 1 0; -1 0 0 1')
        else:
            grad = np.matrix('0 0 0 0; 0 0 0 0; 0 0 0 0')

        return grad
