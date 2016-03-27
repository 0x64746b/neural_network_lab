#!/usr/bin/env python
# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import os

# external dependency [provided by organizer](https://www2.informatik.uni-hamburg.de/~weber/code/KTimage.py).
from KTimage import importimage as import_image
import numpy as np
from scipy.special import expit



NUM_HIDDEN_NODES = 100
ACCEPTED_ERROR = 1e-3
ERROR_RATE = 0.05
LEARNING_RATE = 0.1

# Input samples provided by organizer.
# TODO: Check results for [MNIST data](http://yann.lecun.com/exdb/mnist/)
# TODO: Parse commandline
INPUT_DIR = 'data/digits_alph'


def expit_prime(h):
    return expit(h) * (1 - expit(h))


def softmax(h):
    # TODO: Worth using [Theano's implementation](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.softmax)?
    #       Does the returned `Softmax.0` instance need to be explicitly
    #       `eval()`ed or can it just be passed around?
    return np.exp(h) / np.exp(h).sum()


def get_input_data(files):
    random_index = np.random.randint(0, len(files))

    file_name = input_files[random_index]
    input_data = import_image(os.path.join(INPUT_DIR, file_name))[0]

    expected_value = np.zeros(len(files))
    expected_value[random_index] = 1
    
    return input_data, expected_value


def update_ema(current, average):
    """Update the exponential moving average."""
    return ERROR_RATE * abs(current) + (1 - ERROR_RATE) * average


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
    input_files = sorted(os.listdir(INPUT_DIR))
    input_data, expected_value = get_input_data(input_files)

    hidden = Layer(NUM_HIDDEN_NODES, input_data.shape[0], expit)
    output = Layer(len(input_files), NUM_HIDDEN_NODES, softmax)

    average_errors = np.ones(len(input_files))
    accepted_errors = np.full(len(input_files), ACCEPTED_ERROR)

    while not np.all(np.less(average_errors, accepted_errors)):
        # get a random date
        input_data, expected_value = get_input_data(input_files)

        # process inputs
        outputs = output.process(hidden.process(input_data))

        # calculate errors
        output.errors = expected_value - outputs
        hidden.errors = expit_prime(hidden.h) * np.dot(output.errors, output.weights)

        # update weights and biases
        output.update()
        hidden.update()

        average_errors = update_ema(output.errors, average_errors)

    print('last errors:\n', output.errors)
    print('last outputs:\n', outputs)
    print('last outputs rounded:\n', outputs.round())

    #index = expected_value.nonzero()[0][0]
    #print('index of class:', index)
    #print('output of class:', outputs[index])
    #print('error for class:', output.errors[index])
