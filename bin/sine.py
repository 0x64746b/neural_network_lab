#!/usr/bin/env python
# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)


import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from mlp import Layer, RecurrentLayer


# [0, max, pi, min] + 3 recursive bisections for each resulting quarter
NUM_SAMPLES = 32
NUM_TRAINING_CYCLES = 10000
NUM_EPOCHS = NUM_TRAINING_CYCLES * NUM_SAMPLES

GENERATING_FACTOR = 4
NUM_GENERATED_SAMPLES = GENERATING_FACTOR * NUM_SAMPLES

NUM_TRAINING_FREQUENCIES = 3
GENERATING_FREQUENCIES = [1.0, 2.0]

NUM_HIDDEN_NODES = 30
HISTORY_LENGTH = 3
LEARNING_RATE = 0.01


def expit_prime(h):
    return expit(h) * (1 - expit(h))


if __name__ == '__main__':
    # setup data
    sampling_points = np.linspace(0, 2*np.pi, num=NUM_SAMPLES, endpoint=False)

    # construct net
    hidden = RecurrentLayer(NUM_HIDDEN_NODES, 2, expit, HISTORY_LENGTH, LEARNING_RATE)
    output = Layer(1, NUM_HIDDEN_NODES, lambda x: x, LEARNING_RATE)

    # train
    print('Training...')
    last_training_run = np.zeros(NUM_SAMPLES)
    last_training_errors = np.zeros(NUM_SAMPLES)

    for epoch in range(NUM_EPOCHS):
        current_index = epoch % NUM_SAMPLES
        next_index = (current_index + 1) % NUM_SAMPLES

        if epoch % (20 * NUM_SAMPLES) == 0:
            # change training frequency
            frequency_factor = float(np.random.random_integers(NUM_TRAINING_FREQUENCIES))

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
