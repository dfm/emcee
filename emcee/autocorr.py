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
    f = np.fft.fft(x, axis=axis)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis).real
    return acf/len(acf)


def integrated_time(x, axis=0):
    """
    Estimate the integrated autocorrelation time of a time series.

    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.

    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.

    """
    # Compute the autocorrelation function.
    f = function(x, axis=axis)

    # Compute the integrated autocorrelation time.
    first = [slice(None), ] * len(f.shape)
    first[axis] = 0
    sig2 = f[first]
    N = f.shape[axis]
    tau = 1 + 2*np.sum((1-np.arange(1, len(f))/N)*f[1:]/sig2, axis=axis)

    # if full_output:
    #     return tau, np.mean(x), np.sqrt(sig2*tau/N)
    return tau
