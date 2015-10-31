#!/usr/bin/env
# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import numpy as np
from scipy.special import expit


DIMENSION_HID = 2
# make small, so not easily met accidentally
ACCEPTED_ERROR = 1e-10
LEARNING_RATE = 0.1


expit_prime = lambda h: expit(h) * (1 - expit(h))


class Layer(object):

    """Encapsulate the state of a layer."""

    def __init__(self, dimension, input_dimension, transfer_func):
        self.weights = np.random.uniform(-1.0, 1.0, (dimension, input_dimension))
        self.biases = np.ones(dimension)
        self.errors = np.ones(dimension)
        self._transfer_func = transfer_func

    def process(self, input_vector):
        self.input_vector = input_vector
        self.h = np.dot(self.weights, input_vector) + self.biases
        return self._transfer_func(self.h)

    def backpropagate(self):
        self.weights += LEARNING_RATE * np.outer(self.errors, self.input_vector)
        self.biases += LEARNING_RATE * self.errors


if __name__ == '__main__':
    # inputs (XOR) and expected outputs
    data_in = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    data_out = np.array([[0], [1], [1], [0]])

    hidden = Layer(DIMENSION_HID, data_in.shape[1], expit)
    output = Layer(data_out.shape[1], DIMENSION_HID, lambda x: x)

    while abs(output.errors) > ACCEPTED_ERROR:
        random_index = np.random.randint(0, data_in.shape[0])

        # process inputs
        outputs = output.process(hidden.process(data_in[random_index]))

        # calculate errors
        output.errors = data_out[random_index] - outputs
        hidden.errors = expit_prime(hidden.h) * np.dot(output.errors, output.weights)

        # update weights and biases
        output.backpropagate()
        hidden.backpropagate()

        print('error:', output.errors)

    print('Weights:\n', hidden.weights, output.weights)
