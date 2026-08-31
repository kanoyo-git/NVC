"""Small NumPy helpers shared by pitch-extraction code and its tests."""

import numpy as np


def preserve_unvoiced_f0(f0):
    """Return a finite F0 contour while keeping unvoiced frames at zero."""
    clean = np.asarray(f0, dtype=np.float32).copy()
    clean[~np.isfinite(clean) | (clean <= 0)] = 0.0
    return clean


def interpolate_unvoiced_f0(f0):
    """Compatibility alias retained for callers from older NVC revisions."""
    return preserve_unvoiced_f0(f0)
