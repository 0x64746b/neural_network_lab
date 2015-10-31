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


DIMENSION_HID = 3
NUM_REPETITIONS = 10000
LEARNING_RATE = 0.1


expit_prime = lambda h: expit(h) * (1 - expit(h))


if __name__ == '__main__':
    # inputs (XOR) and expected outputs
    data_in = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    data_out = np.array([[0], [1], [1], [0]])

    dimension_in = data_in.shape[1]
    dimension_out = data_out.shape[1]

    # initialize weights
    W_hid = np.random.uniform(-1.0, 1.0, (DIMENSION_HID, dimension_in))
    W_out = np.random.uniform(-1.0, 1.0, (dimension_out, DIMENSION_HID))

    # initialize biases
    b_hid = np.ones(DIMENSION_HID)
    b_out = np.ones(dimension_out)

    for i in range(NUM_REPETITIONS):
        random_index = np.random.randint(0, data_in.shape[0])

        #print('W_hid:\n', W_hid)
        #print('W_out:\n', W_out)

        #print('b_hid:', b_hid)
        #print('b_out:', b_out)

        # forward propagation
        s_in = data_in[random_index]
        #print('s_in:', s_in)

        h_hid = np.dot(W_hid, s_in) + b_hid
        s_hid = expit(h_hid)
        #print('s_hid:', s_hid)

        h_out = np.dot(W_out, s_hid) + b_out
        s_out = h_out
        #print('s_out:', s_out)

        # calculate error
        error_out = data_out[random_index] - s_out
        print('error_out:', error_out)

        # backpropagation
        error_hid = expit_prime(h_hid) * np.dot(error_out, W_out)
        #print('error_hid:', error_hid)

        W_out_delta = LEARNING_RATE * np.outer(error_out, s_hid)
        W_hid_delta = LEARNING_RATE * np.outer(error_hid, s_in)
        #print('W_out_delta:\n', W_out_delta)
        #print('W_hid_delta:\n', W_hid_delta)

        W_out += W_out_delta
        W_hid += W_hid_delta
        #print('Adapted W_out:\n', W_out)
        #print('Adapted W_hid:\n', W_hid)

        b_out_delta = LEARNING_RATE * error_out
        b_hid_delta = LEARNING_RATE * error_hid
        #print('b_out_delta:', b_out_delta)
        #print('b_hid_delta:', b_hid_delta)

        b_out += b_out_delta
        b_hid += b_hid_delta
        #print('Adapted b_out:', b_out)
        #print('Adapted b_hid:', b_hid)

        #print('\n===================\n')
