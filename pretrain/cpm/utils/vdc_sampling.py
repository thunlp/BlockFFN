from functools import partial

import matplotlib.pyplot as plt
import numpy as np


def van_der_corput(n, base=2):
    """Generate the n-th value in the Van der Corput sequence."""
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return vdc


def van_der_corput_sampling_gen(vdc_values):
    """Generator function for sampling indices based on weights using the Van der Corput sequence."""

    def gen(weights, vdc_value):
        cdf = np.cumsum(weights)
        sample = np.searchsorted(cdf, vdc_value)
        return sample

    sample_index = 0
    # Pre-generate Van der Corput sequence
    max_samples = 100000  # or any number that you find suitable

    while True:
        # Generate the next value in the Van der Corput sequence
        vdc_value = vdc_values[sample_index % max_samples]
        # Generate a sample index based on the Van der Corput value and the CDF
        yield partial(gen, vdc_value=vdc_value)
        sample_index += 1
