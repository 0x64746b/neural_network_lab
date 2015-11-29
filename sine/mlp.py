#!/usr/bin/env python
# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)


import math

import numpy as np
from scipy.special import expit

# [0, max, pi, min] + 3 recursive bisections for each resulting quarter
NUM_SAMPLES = 4 + 4 * 7

NUM_HIDDEN_NODES = 5
ACCEPTED_ERROR = 1e-3
ERROR_RATE = 0.05
LEARNING_RATE = 0.1


def expit_prime(h):
    return expit(h) * (1 - expit(h))


class Layer(object):
    """Encapsulate the state of a layer."""

    def __init__(self, dimension, input_dimension, transfer_func):
        self.weights = np.random.uniform(-1.0, 1.0, (dimension, input_dimension))
        self.biases = np.ones(dimension)
        self.errors = np.ones(dimension)

        self._input_vector = np.zeros(input_dimension)
        self._transfer_func = transfer_func

    def process(self, input_vector):
        self._input_vector = input_vector
        # FIXME: Limited to transfer functions that work on the weighted sum of
        #        the inputs
        self.h = np.dot(self.weights, input_vector) + self.biases
        return self._transfer_func(self.h)

    def update(self):
        self.weights += LEARNING_RATE * np.outer(self.errors, self._input_vector)
        self.biases += LEARNING_RATE * self.errors


class RecurrentLayer(Layer):
    """Encapsulate the state of a recurrent layer."""

    def __init__(self, dimension, input_dimension, transfer_func):
        self.recurrent_weights = np.random.uniform(-1.0, 1.0, (dimension, dimension))
        self._previous_output = np.zeros(dimension)

        super(RecurrentLayer, self).__init__(dimension, input_dimension, transfer_func)

    def process(self, input_vector):
        self.h = np.dot(self.weights, input_vector) + np.dot(self.recurrent_weights, self._previous_output) + self.biases

        # remember output for next round
        self._previous_output = self._transfer_func(self.h)

        return self._previous_output

    def update(self):
        self.recurrent_weights = LEARNING_RATE * np.outer(self.errors, self._previous_output)

        super(RecurrentLayer, self).update()


if __name__ == '__main__':
    sampling_points = np.linspace(0, 2*math.pi, num=NUM_SAMPLES, endpoint=False)
    input_data = np.array([math.sin(x) for x in sampling_points]).reshape((NUM_SAMPLES, 1))

    hidden = RecurrentLayer(NUM_HIDDEN_NODES, input_data.shape[1], expit)
    output = Layer(input_data.shape[1], NUM_HIDDEN_NODES, lambda x: x)

    for epoch in range(100 * NUM_SAMPLES):
        current_index = epoch % NUM_SAMPLES
        next_index = (current_index + 1) % NUM_SAMPLES

        # process inputs
        outputs = output.process(hidden.process(input_data[current_index]))

        # calculate errors
        output.errors = input_data[next_index] - outputs
        hidden.errors = expit_prime(hidden.h) * np.dot(output.errors, output.weights)

        # update weights and biases
        output.update()
        hidden.update()

        print('input:', input_data[current_index])
        print('expected output:', input_data[next_index])
        print('actual output:', outputs)
        print('output error:', output.errors)
