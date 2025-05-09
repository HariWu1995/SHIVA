import numpy as np


def blend_h(
    a: np.ndarray, 
    b: np.ndarray, 
    extent: int,
):
    blend_extent = min(a.shape[1], b.shape[1], extent)
    for x in range(blend_extent):
        b[:, x, :] = \
        b[:, x, :]                 * (    x / blend_extent) + \
        a[:, -blend_extent + x, :] * (1 - x / blend_extent)
    return b

