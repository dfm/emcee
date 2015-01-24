# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["function", "integrated_time"]

import numpy as np


def function(x, axis=0, fast=False):
    """
    Estimate the autocorrelation function of a time series using the FFT.

    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.

    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.

    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)

    """
    x = np.atleast_1d(x)
    m = [slice(None), ] * len(x.shape)

    # For computational efficiency, crop the chain to the largest power of
    # two if requested.
    if fast:
        n = int(2**np.floor(np.log2(x.shape[axis])))
        m[axis] = slice(0, n)
        x = x
    else:
        n = x.shape[axis]

    # Compute the FFT and then (from that) the auto-correlation function.
    f = np.fft.fft(x-np.mean(x, axis=axis), n=2*n, axis=axis)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[m].real
    m[axis] = 0
    return acf / acf[m]


def integrated_time(x, axis=0, window=50, fast=False):
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
        The size of the window to use. (default: 50)

    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)

    """
    # Compute the autocorrelation function.
    f = function(x, axis=axis, fast=fast)

    # Special case 1D for simplicity.
    if len(f.shape) == 1:
        return 1 + 2*np.sum(f[1:window])

    # N-dimensional case.
    m = [slice(None), ] * len(f.shape)
    m[axis] = slice(1, window)
    tau = 1 + 2*np.sum(f[m], axis=axis)

    return tau


def safe_integrated_time(x, c=10, M=10, maxiter=None, dM=1, axis=0,
                         fast=False):
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
        The size of the window to use. (default: 50)

    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)

    """
    M = int(M)
    size = 0.5*x.shape[axis]
    if M >= size:
        raise AutocorrError("The chain is too short")

    # Compute the autocorrelation function.
    f = function(x, axis=axis, fast=fast)

    # Check the dimensions of the array.
    oned = len(f.shape) == 1
    m = [slice(None), ] * len(f.shape)

    # Loop over proposed window sizes until convergence is reached.
    tau = None
    i = 0
    while tau is None or M <= c * tau.max():
        if oned:
            # Special case 1D for simplicity.
            tau = 1 + 2*np.sum(f[1:M])
        else:
            # N-dimensional case.
            m[axis] = slice(1, M)
            tau = 1 + 2*np.sum(f[m], axis=axis)

        # Check the size of the array and make sure that it's lone enough.
        if c * tau.max() >= size:
            raise AutocorrError("The chain is too short to reliably estimate "
                                "the autocorrelation time")

        if maxiter is not None and i >= maxiter:
            break
        i += 1
        M += dM
    return tau


class AutocorrError(Exception):
    pass


if __name__ == "__main__":
    import time
    # import acor

    N = 2000
    a = 0.9
    d = 3
    x = np.empty((N, d))
    x[0] = np.zeros(d)
    for i in xrange(1, N):
        x[i] = x[i-1] * a + np.random.rand(d)

    strt = time.time()
    print(safe_integrated_time(x))
    print(time.time() - strt)

    # strt = time.time()
    # print([acor.acor(x[:, i])[0] for i in range(d)])
    # print(time.time() - strt)
