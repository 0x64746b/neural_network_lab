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

from mlp import expit_prime, Layer, softmax


NUM_HIDDEN_NODES = 100
ACCEPTED_ERROR = 1e-3
ERROR_RATE = 0.05
LEARNING_RATE = 0.1

# Input samples provided by organizer.
# TODO: Check results for [MNIST data](http://yann.lecun.com/exdb/mnist/)
# TODO: Parse commandline
INPUT_DIR = 'data/digits_alph'


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
