# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["function", "integrated_time", "AutocorrError"]

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


def integrated_time(x, low=10, high=None, step=1, c=10, full_output=False,
                    axis=0, fast=False):
    """
    Estimate the integrated autocorrelation time of a time series using the
    iterative procedure described on page 16 of `Sokal's notes
    <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ to determine a
    reasonable window size.

    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.

    :param low: (optional)
        The minimum window size to test. (default: ``10``)

    :param high: (optional)
        The maximum window size to test. (default: ``x.shape[axis] / (2*c)``)

    :param step: (optional)
        The step size for the window search. (default: ``1``)

    :param full_output: (optional)
        Return the final window size as well as the autocorrelation time.
        (default: ``False``)

    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.

    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)

    :return tau:
        An estimate of the integrated autocorrelation time of the time series
        ``x`` computed along the axis ``axis``.

    :return window_size: (optional)
        The final window size that was used. Only returned if ``full_output``
        is ``True``.

    :raises AutocorrError:
        If the autocorrelation time can't be reliably estimated from the
        chain. This normally means that the chain is too short.

    """
    size = 0.5*x.shape[axis]
    if int(c * low) >= size:
        raise AutocorrError("The chain is too short")

    # Compute the autocorrelation function.
    f = function(x, axis=axis, fast=fast)

    # Check the dimensions of the array.
    oned = len(f.shape) == 1
    m = [slice(None), ] * len(f.shape)

    # Loop over proposed window sizes until convergence is reached.
    if high is None:
        high = int(size / c)
    for M in np.arange(low, high, step).astype(int):
        # Compute the autocorrelation time with the given window.
        if oned:
            # Special case 1D for simplicity.
            tau = 1 + 2*np.sum(f[1:M])
        else:
            # N-dimensional case.
            m[axis] = slice(1, M)
            tau = 1 + 2*np.sum(f[m], axis=axis)

        # Accept the window size if it satisfies the convergence criterion.
        if M > c * tau.max():
            if full_output:
                return tau, M
            return tau

        # If the autocorrelation time is too long to be estimated reliably
        # from the chain, it should fail.
        if c * tau.max() >= size:
            break

    raise AutocorrError("The chain is too short to reliably estimate "
                        "the autocorrelation time")


class AutocorrError(Exception):
    """
    Raised by :func:`integrated_time` if the chain is too short to estimate
    an autocorrelation time.

    """
    pass


if __name__ == "__main__":
    import time

    N = 5000000
    a = 0.9
    d = 3
    x = np.empty((N, d))
    x[0] = np.zeros(d)
    for i in xrange(1, N):
        x[i] = x[i-1] * a + np.random.rand(d)

    strt = time.time()
    print(integrated_time(x))
    print(time.time() - strt)
