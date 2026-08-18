"""Small NumPy helpers shared by pitch-extraction code and its tests."""

import numpy as np


def interpolate_unvoiced_f0(f0):
    """Fill F0 gaps while preserving a completely unvoiced segment as zeros."""
    f0 = np.asarray(f0)
    unvoiced = f0 <= 0
    voiced = ~unvoiced
    if unvoiced.any() and voiced.any():
        f0[unvoiced] = np.interp(
            np.flatnonzero(unvoiced), np.flatnonzero(voiced), f0[voiced]
        )
    return f0
