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

    if len(sys.argv) != 6:
        sys.exit(
            'USAGE: $ python two_dots <FIRST_SAMPLE> <FIRST_PREDICTION>'
            '<SECOND_HISTORY> <SECOND_SAMPLE> <SECOND_PREDICTION>')

    first_sample, first_prediction = int(sys.argv[1]), int(sys.argv[2])
    second_history, second_sample, second_prediction = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

    sampling_points = np.linspace(0, 2*np.pi, num=NUM_SAMPLES)
    sine_input = np.sin(FREQUENCY_FACTOR * sampling_points)

    plt.plot(sampling_points, sine_input, 'b')
    plt.plot(2*np.pi, 0, 'ko')

    plt.plot(
        sampling_points[first_sample],
        np.sin(sampling_points[first_sample]),
        'D',
        color='0.85'
    )
    plt.plot(
        sampling_points[first_prediction],
        np.sin(sampling_points[first_prediction]),
        'D',
        color='0.85'
    )

    plt.plot(
        sampling_points[second_history],
        np.sin(sampling_points[second_history]),
        'rD'
    )
    plt.plot(
        sampling_points[second_sample],
        np.sin(sampling_points[second_sample]),
        'bD'
    )
    plt.plot(
        sampling_points[second_prediction],
        np.sin(sampling_points[second_prediction]),
        'gD'
    )
    plt.annotate(
        '!',
        xy=(sampling_points[second_prediction], np.sin(sampling_points[second_prediction])),
        xytext=(sampling_points[second_prediction]+0.15, np.sin(sampling_points[second_prediction]))
    )

    plt.axhline(color='k')
    plt.savefig(
        'five_dots_{}_{}_{}_{}_{}.png'.format(
            first_sample,
            first_prediction,
            second_history,
            second_sample,
            second_prediction,
        )
    )
