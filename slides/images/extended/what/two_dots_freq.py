#!/usr/bin/env python
# coding: utf-8

from __future__ import(
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import sys

from matplotlib import pyplot as plt
import numpy as np


NUM_SAMPLES = 33

FIRST_DOT_INDEX = 12
SECOND_DOT_INDEX = FIRST_DOT_INDEX + 1

if __name__ == '__main__':

    if len(sys.argv) != 4:
        sys.exit('USAGE: $ python two_dots <FREQUENCY> <FIRST_SAMPLE> <SECOND_SAMPLE>')

    frequency_factor, first_sample, second_sample = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    sampling_points = np.linspace(0, 2*np.pi, num=NUM_SAMPLES)
    sine_input = np.sin(frequency_factor * sampling_points)

    plt.plot(sampling_points, sine_input, 'b', label='Frequency: {}'.format(frequency_factor))
    plt.plot(2*np.pi, 0, 'ko')

    plt.plot(
        sampling_points[first_sample],
        sine_input[first_sample],
        'bD'
    )
    plt.plot(
        sampling_points[second_sample],
        sine_input[second_sample],
        'bD'
    )

    plt.axhline(color='k')
    plt.legend()
    plt.savefig('two_dots_freq_{}_{}_{}.png'.format(frequency_factor, first_sample, second_sample))
