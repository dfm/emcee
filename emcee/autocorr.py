#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["function", "integrated_time"]

import numpy as np


def function(x, axis=0):
    """
    Estimate the autocorrelation function of a time series using the FFT.

    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.

    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.

    """
    x = np.atleast_1d(x)
    n = x.shape[axis]
    f = np.fft.fft(x-np.mean(x, axis=axis), n=2*n, axis=axis)
    m = [slice(None), ] * len(x.shape)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[m].real
    m[axis] = 0
    return acf / acf[m]


def integrated_time(x, axis=0, window=50):
    """
    Estimate the integrated autocorrelation time of a time series.

    See `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ on
    MCMC and sample estimators for autocorrelation times.

    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.

    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.

    :param window: (optional)
        The size of the window to use.

    """
    # Compute the autocorrelation function.
    f = function(x, axis=axis)

    # Special case 1D for simplicity.
    if len(f.shape) == 1:
        return 1 + 2*np.sum(f[1:window])

    # N-dimensional case.
    m = [slice(None), ] * len(f.shape)
    m[axis] = slice(1, window)
    tau = 1 + 2*np.sum(f[m], axis=axis)

    return tau


if __name__ == "__main__":
    import time
    import acor

    N = 400000
    a = 0.9
    d = 3
    x = np.empty((N, d))
    x[0] = np.zeros(d)
    for i in xrange(1, N):
        x[i] = x[i-1] * a + np.random.rand(d)

    strt = time.time()
    print(integrated_time(x))
    print(time.time() - strt)

    strt = time.time()
    print([acor.acor(x[:, i])[0] for i in range(d)])
    print(time.time() - strt)
