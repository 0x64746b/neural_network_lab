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
FREQUENCY_FACTOR = 1

FIRST_DOT_INDEX = 12
SECOND_DOT_INDEX = FIRST_DOT_INDEX + 1

if __name__ == '__main__':

    if len(sys.argv) != 3:
        sys.exit('USAGE: $ python two_dots <FIRST_SAMPLE> <SECOND_SAMPLE>')

    first_sample, second_sample = int(sys.argv[1]), int(sys.argv[2])

    sampling_points = np.linspace(0, 2*np.pi, num=NUM_SAMPLES)
    sine_input = np.sin(FREQUENCY_FACTOR * sampling_points)

    plt.plot(sampling_points, sine_input, 'b')
    plt.plot(2*np.pi, 0, 'ko')

    plt.plot(
        sampling_points[first_sample],
        np.sin(sampling_points[first_sample]),
        'bD'
    )
    plt.plot(
        sampling_points[second_sample],
        np.sin(sampling_points[second_sample]),
        'bD'
    )

    plt.axhline(color='k')
    plt.savefig('two_dots_{}_{}.png'.format(first_sample, second_sample))
