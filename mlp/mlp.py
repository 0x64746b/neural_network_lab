# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)


from collections import deque

import numpy as np
from scipy.special import expit


def expit_prime(h):
    return expit(h) * (1 - expit(h))


class Layer(object):
    """Encapsulate the state of a layer."""

    def __init__(self, dimension, input_dimension, transfer_func, learning_rate=0.1):
        self.weights = np.random.uniform(-1.0, 1.0, (dimension, input_dimension))
        self.biases = np.ones(dimension)
        self.errors = np.ones(dimension)

        self._input_vector = np.zeros(input_dimension)
        self._transfer_func = transfer_func
        self._learning_rate = learning_rate

    def process(self, input_vector):
        self._input_vector = input_vector
        # FIXME: Limited to transfer functions that work on the weighted sum of
        #        the inputs
        self.h = np.dot(self.weights, input_vector) + self.biases
        return self._transfer_func(self.h)

    def update(self):
        self.weights += self._learning_rate * np.outer(self.errors, self._input_vector)
        self.biases += self._learning_rate * self.errors


class RecurrentLayer(Layer):
    """Encapsulate the state of a recurrent layer."""

    def __init__(self, dimension, input_dimension, transfer_func, history_length=None, learning_rate=0.1):
        super(RecurrentLayer, self).__init__(dimension, input_dimension, transfer_func, learning_rate)

        self.recurrent_weights = np.random.uniform(-1.0, 1.0, (dimension, dimension))
        self.input_vectors = deque(maxlen=history_length)
        self.h = deque(maxlen=history_length)
        self.outputs = deque(maxlen=history_length)
        self.outputs.append(np.zeros(dimension)) 
        self.errors = deque(maxlen=history_length)

        self._dimension = dimension

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
            self.weights += self._learning_rate * np.outer(self.errors[index], self.input_vectors[index])
            try:
                self.recurrent_weights += self._learning_rate * np.outer(self.errors[index], self.outputs[index+1])
            except IndexError:
                pass
            self.biases += self._learning_rate * self.errors[index]

    def clear(self):
        self.input_vectors.clear()
        self.h.clear()
        self.outputs.clear()
        self.errors.clear()

        self.outputs.append(np.zeros(self._dimension))