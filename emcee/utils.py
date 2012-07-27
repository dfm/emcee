__all__ = ["sample_ball"]


import numpy as np


def sample_ball(p0, std, size=1):
    """
    Produce a ball of walkers around an initial parameter value.

    :param p0: The initial parameter value.
    :param std: The axis-aligned standard deviation.
    :param size: The number of samples to produce.

    """
    assert(len(p0) == len(std))
    return np.vstack([p0 + std * np.random.normal(size=len(p0))
                        for i in range(size)])
