#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["sample_ball", "MH_proposal_axisaligned", "MPIPool"]


import numpy as np

from .mpi_pool import MPIPool


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


class MH_proposal_axisaligned(object):
    """
    A Metropolis-Hastings proposal, with axis-aligned Gaussian steps,
    for convenient use as the ``mh_proposal`` option to
    :func:`EnsembleSampler.sample` .

    """
    def __init__(self, stdev):
        self.stdev = stdev

    def __call__(self, X):
        (nw, npar) = X.shape
        assert(len(self.stdev) == npar)
        return X + self.stdev * np.random.normal(size=X.shape)
