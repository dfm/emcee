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

    # Special case 1D for simplicity.
    if len(f.shape) == 1:
        N = len(f)
        return 1 + 2*np.sum((1-np.arange(1, N)/N)*f[1:]/f[0])

    # Estimate the variance along the time axis.
    m = [slice(None), ] * len(f.shape)
    m[axis] = 0
    sig2 = f[m]

    # Compute the broadcasting mask for the time array.
    bc = [np.newaxis, ] * len(f.shape)
    bc[axis] = slice(None)

    # Compute the broadcasting mask for the variance array.
    var_bc = [slice(None), ] * len(f.shape)
    var_bc[axis] = np.newaxis

    # Compute the integrated autocorrelation time.
    N = f.shape[axis]
    m[axis] = slice(1, N)
    tau = 1 + 2*np.sum((1-np.arange(1, N)/N)[bc]*f[m]/sig2[var_bc], axis=axis)

    return tau
