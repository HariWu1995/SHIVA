import numpy as np


def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

