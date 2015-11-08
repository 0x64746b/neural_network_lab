#!/usr/bin/env python
# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import numpy as np
from pandas import ewma
from scipy.special import expit


NUM_HIDDEN_NODES = 3
ACCEPTED_ERROR = 1e-10
ERROR_RATE = 0.05
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
        # FIXME: Limited to transfer functions that work on the weighted sum of
        #        the inputs
        self.h = np.dot(self.weights, input_vector) + self.biases
        return self._transfer_func(self.h)

    def update(self):
        self.weights += LEARNING_RATE * np.outer(self.errors, self.input_vector)
        self.biases += LEARNING_RATE * self.errors


if __name__ == '__main__':
    # inputs (XOR) and expected outputs
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_outputs = np.array([[0], [1], [1], [0]])

    hidden = Layer(NUM_HIDDEN_NODES, input_data.shape[1], expit)
    output = Layer(expected_outputs.shape[1], NUM_HIDDEN_NODES, lambda x: x)

    average_error = 1
    while average_error > ACCEPTED_ERROR:
        random_index = np.random.randint(0, input_data.shape[0])

        # process inputs
        outputs = output.process(hidden.process(input_data[random_index]))

        # calculate errors
        output.errors = expected_outputs[random_index] - outputs
        hidden.errors = expit_prime(hidden.h) * np.dot(output.errors, output.weights)

        # update weights and biases
        output.update()
        hidden.update()

        average_error = ewma(
            np.array([average_error, abs(output.errors)]),
            com=(1 / ERROR_RATE - 1),
            adjust=False
        )[-1]
        print('Error:', output.errors, 'Avg:', average_error)

    print('Weights:\n', hidden.weights, output.weights)
