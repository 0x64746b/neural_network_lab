#!/usr/bin/env python
# coding: utf-8


from matplotlib import pyplot as plt
import numpy as np


NUM_SAMPLES = 33
FREQUENCY_FACTOR = 1


if __name__ == '__main__':

    sampling_points = np.linspace(0, 2*np.pi, num=NUM_SAMPLES)
    sine_input = np.sin(FREQUENCY_FACTOR * sampling_points)

    plt.plot(sampling_points, sine_input, 'b')
    plt.plot(2*np.pi, 0, 'ko')
    plt.plot(sampling_points[12], np.sin(sampling_points[12]), 'bD')

    plt.axhline(color='k')
    plt.savefig('one_dot.png')
