# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_autocorr"]

import numpy as np
from ...compat import xrange
from ...autocorr import integrated_time


def test_autocorr(seed=1234, ndim=3, N=100000):
    np.random.seed(seed)
    a = 0.9
    x = np.empty((N, ndim))
    x[0] = np.zeros(ndim)
    for i in xrange(1, N):
        x[i] = x[i-1] * a + np.random.rand(ndim)
    tau = integrated_time(x)
    assert np.all(np.abs(tau - 19.0) / 19. < 0.2)
