#!/usr/bin/env python
# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)


from collections import deque
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

# [0, max, pi, min] + 3 recursive bisections for each resulting quarter
NUM_SAMPLES = 4 + 4 * 7
NUM_EPOCHS = 50000 * NUM_SAMPLES
HISTORY_LENGTH = 3

NUM_HIDDEN_NODES = 30
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

    def __init__(self, dimension, input_dimension, transfer_func, history_length=None):
        super(RecurrentLayer, self).__init__(dimension, input_dimension, transfer_func)

        self.recurrent_weights = np.random.uniform(-1.0, 1.0, (dimension, dimension))
        self.input_vectors = deque(maxlen=history_length)
        self.h = deque(maxlen=history_length)
        self.outputs = deque(np.zeros(dimension * dimension).reshape(dimension, dimension), maxlen=history_length)
        self.errors = deque(maxlen=history_length)

    def process(self, input_vector):
        weighted_sum = np.dot(self.weights, input_vector) + np.dot(self.recurrent_weights, self.outputs[0]) + self.biases
        activation = self._transfer_func(weighted_sum)

        # update history
        self.input_vectors.appendleft(input_vector)
        self.h.appendleft(weighted_sum)
        self.outputs.appendleft(activation)

        return activation

    def update(self):
        for index in range(len(self.errors)):
            self.weights += LEARNING_RATE * np.outer(self.errors[index], self.input_vectors[index])
            try:
                self.recurrent_weights += LEARNING_RATE * np.outer(self.errors[index], self.outputs[index+1])
            except IndexError:
                pass
            self.biases += LEARNING_RATE * self.errors[index]


if __name__ == '__main__':
    sampling_points = np.linspace(0, 2*math.pi, num=NUM_SAMPLES, endpoint=False)
    input_data = np.array([math.sin(x) for x in sampling_points]).reshape((NUM_SAMPLES, 1))
    last_training_run = np.zeros(NUM_SAMPLES)
    last_training_errors = np.zeros(NUM_SAMPLES)

    hidden = RecurrentLayer(NUM_HIDDEN_NODES, input_data.shape[1], expit, HISTORY_LENGTH)
    output = Layer(input_data.shape[1], NUM_HIDDEN_NODES, lambda x: x)

    for epoch in range(NUM_EPOCHS):
        current_index = epoch % NUM_SAMPLES
        next_index = (current_index + 1) % NUM_SAMPLES

        # process inputs
        outputs = output.process(hidden.process(input_data[current_index]))

        # backpropagate errors
        output.errors = input_data[next_index] - outputs

        hidden.errors.appendleft(expit_prime(hidden.h[0]) * np.dot(output.errors, output.weights))
        for index in range(1, len(hidden.errors)):
            hidden.errors[index] = expit_prime(hidden.h[index]) * np.dot(hidden.errors[index-1], hidden.weights)

        # learn
        output.update()
        hidden.update()

        if epoch > (NUM_EPOCHS - NUM_SAMPLES):
            print('input:', input_data[current_index])
            print('expected output:', input_data[next_index])
            print('actual output:', outputs)
            print('output error:', output.errors[0])
            last_training_run[current_index] = outputs
            last_training_errors[current_index] = output.errors

    # plot results
    plt.plot(sampling_points, input_data, 'b', label='input')
    plt.plot(sampling_points, last_training_run, 'r', label='last training run')
    plt.plot(sampling_points, last_training_errors, '0.5', label='errors')

    plt.axhline(color='k')
    plt.legend()
    plt.show()
