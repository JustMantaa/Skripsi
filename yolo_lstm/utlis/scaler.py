import numpy as np


def apply_scaler(x, mean, scale):
    if mean is None:
        return x

    n, t, f = x.shape
    flat = x.reshape(n, t * f)
    scale_safe = np.where(scale == 0, 1.0, scale)
    flat = (flat - mean) / scale_safe
    return flat.reshape(n, t, f)
