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

from mlp import Layer

NUM_HIDDEN_NODES = 3
ACCEPTED_ERROR = 1e-10
ERROR_RATE = 0.05
LEARNING_RATE = 0.1


expit_prime = lambda h: expit(h) * (1 - expit(h))


class AverageError(object):

    def __init__(self, target, smoothing_factor):
        self._target = target
        self._center_of_mass = 1 / smoothing_factor - 1
        self._value = 1.0

    def update(self, error):
        self._value = ewma(
            np.array([self._value, abs(error)]),
            com=self._center_of_mass,
            adjust=False
        )[-1]

    @property
    def too_large(self):
        return bool(self._value > self._target)

    def __str__(self):
        return str(self._value)


if __name__ == '__main__':
    # inputs (XOR) and expected outputs
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_outputs = np.array([[0], [1], [1], [0]])

    hidden = Layer(NUM_HIDDEN_NODES, input_data.shape[1], expit)
    output = Layer(expected_outputs.shape[1], NUM_HIDDEN_NODES, lambda x: x)

    average_error = AverageError(ACCEPTED_ERROR, ERROR_RATE)
    while average_error.too_large:
        random_index = np.random.randint(0, input_data.shape[0])

        # process inputs
        outputs = output.process(hidden.process(input_data[random_index]))

        # calculate errors
        output.errors = expected_outputs[random_index] - outputs
        hidden.errors = expit_prime(hidden.h) * np.dot(output.errors, output.weights)

        # update weights and biases
        output.update()
        hidden.update()

        average_error.update(output.errors)
        print('Error:', output.errors, 'Avg:', average_error)

    print('Weights:\n', hidden.weights, output.weights)
