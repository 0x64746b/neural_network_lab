#!/usr/bin/env python
# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)


from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

# [0, max, pi, min] + 3 recursive bisections for each resulting quarter
NUM_SAMPLES = 32
NUM_TRAINING_CYCLES = 10000
NUM_EPOCHS = NUM_TRAINING_CYCLES * NUM_SAMPLES

GENERATING_FACTOR = 4
NUM_GENERATED_SAMPLES = GENERATING_FACTOR * NUM_SAMPLES
GENERATING_FREQUENCIES = [1.0, 2.0]

NUM_HIDDEN_NODES = 30
HISTORY_LENGTH = 3

ACCEPTED_ERROR = 1e-3
ERROR_RATE = 0.05
LEARNING_RATE = 0.01


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
        self.outputs = deque(np.zeros(dimension), maxlen=history_length)
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
            self.weights += LEARNING_RATE * np.outer(self.errors[index], self.input_vectors[index])
            try:
                self.recurrent_weights += LEARNING_RATE * np.outer(self.errors[index], self.outputs[index+1])
            except IndexError:
                pass
            self.biases += LEARNING_RATE * self.errors[index]

    def clear(self):
        self.input_vectors.clear()
        self.h.clear()
        self.outputs.clear()
        self.errors.clear()

        self.outputs.append(np.zeros(self._dimension))


if __name__ == '__main__':
    # setup data
    sampling_points = np.linspace(0, 2*np.pi, num=NUM_SAMPLES, endpoint=False)

    # construct net
    hidden = RecurrentLayer(NUM_HIDDEN_NODES, 2, expit, HISTORY_LENGTH)
    output = Layer(1, NUM_HIDDEN_NODES, lambda x: x)

    # train
    print('Training...')
    last_training_run = np.zeros(NUM_SAMPLES)
    last_training_errors = np.zeros(NUM_SAMPLES)

    for epoch in range(NUM_EPOCHS):
        current_index = epoch % NUM_SAMPLES
        next_index = (current_index + 1) % NUM_SAMPLES

        if epoch % (20 * NUM_SAMPLES) == 0:
            # change training frequency
            frequency_factor = float(np.random.random_integers(4))

            sine_input = np.sin(frequency_factor * sampling_points)
            frequency_input = [frequency_factor] * NUM_SAMPLES

            input_data = np.insert(
                frequency_input,
                range(NUM_SAMPLES),
                sine_input
            ).reshape(NUM_SAMPLES, 2)

            hidden.clear()

        # process inputs
        outputs = output.process(hidden.process(input_data[current_index]))

        # backpropagate errors
        output.errors = sine_input[next_index] - outputs

        hidden.errors.appendleft(expit_prime(hidden.h[0]) * np.dot(output.errors, output.weights))
        for index in range(1, len(hidden.errors)):
            hidden.errors[index] = expit_prime(hidden.h[index]) * np.dot(hidden.errors[index-1], hidden.recurrent_weights)

        # learn
        output.update()
        hidden.update()

        # log last run
        if epoch >= (NUM_EPOCHS - NUM_SAMPLES):
            last_training_run[next_index] = outputs
            last_training_errors[next_index] = output.errors

    # generate
    color = 0.2
    for frequency in GENERATING_FREQUENCIES:
        print('Generating...')
        generating_run = np.zeros(NUM_GENERATED_SAMPLES)
        current_value = np.array([0.0, frequency])

        hidden.clear()

        for index in range(NUM_GENERATED_SAMPLES):
            next_value = output.process(hidden.process(current_value))
            current_value = np.insert(next_value, 1, frequency)
            generating_run[(index + 1) % NUM_GENERATED_SAMPLES] = next_value

        plt.plot(
            np.linspace(
                0,
                GENERATING_FACTOR*2*np.pi,
                num=NUM_GENERATED_SAMPLES,
                endpoint=False
            ),
            generating_run, str(color),
            label='generated'
        )
        color += 0.4

    # plot results
    print('{:^18} | {:^18} | {:^18} | {:^18}'.format('input', 'expected', 'actual', 'error'))
    print('{:-^18} | {:-^18} | {:-^18} | {:-^18}'.format('', '', '', ''))
    for index in range(NUM_SAMPLES):
        next_index = (index + 1) % NUM_SAMPLES
        print(
            '{:18} | {:18} | {:< 18} | {:< 18}'.format(
                sine_input[index],
                sine_input[next_index],
                last_training_run[next_index],
                last_training_errors[next_index]
            )
        )

    plt.plot(sampling_points, sine_input, 'b', marker='.', label='input')
    plt.plot(sampling_points, last_training_run, 'r', label='learnt')
    plt.plot(sampling_points, last_training_errors, '0.5', label='error')
    plt.plot([2*np.pi, 4*np.pi, 6*np.pi], [0, 0, 0], 'ok')

    plt.axis([0, GENERATING_FACTOR*2*np.pi, -1.5, 1.5])
    plt.axhline(color='k')
    plt.legend()
    plt.show()
