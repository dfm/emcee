# -*- coding: utf-8 -*-

import numpy as np
import pytest

from emcee.autocorr import AutocorrError, integrated_time


def get_chain(seed=1234, ndim=3, N=100000):
    np.random.seed(seed)
    a = 0.9
    x = np.empty((N, ndim))
    x[0] = np.zeros(ndim)
    for i in range(1, N):
        x[i] = x[i - 1] * a + np.random.rand(ndim)
    return x


def test_1d(seed=1234, ndim=1, N=250000):
    x = get_chain(seed=seed, ndim=ndim, N=N)
    tau = integrated_time(x)
    assert np.all(np.abs(tau - 19.0) / 19.0 < 0.2)


def test_nd(seed=1234, ndim=3, N=150000):
    x = get_chain(seed=seed, ndim=ndim, N=N)
    tau = integrated_time(x)
    assert np.all(np.abs(tau - 19.0) / 19.0 < 0.2)


def test_nd_without_walkers(seed=1234, ndim=3, N=10000):
    x = get_chain(seed=seed, ndim=ndim, N=N)
    tau1 = integrated_time(x[:, np.newaxis])
    tau2 = integrated_time(x, has_walkers=False)
    assert np.allclose(tau1, tau2)


def test_too_short(seed=1234, ndim=3, N=100):
    x = get_chain(seed=seed, ndim=ndim, N=N)
    with pytest.raises(AutocorrError):
        integrated_time(x)
    tau = integrated_time(x, quiet=True)  # NOQA


def test_autocorr_multi_works():
    np.random.seed(42)
    xs = np.random.randn(16384, 2)

    acls_multi = integrated_time(xs[:, np.newaxis])
    acls_single = np.array(
        [integrated_time(xs[:, i]) for i in range(xs.shape[1])]
    ).squeeze()

    assert np.allclose(acls_multi, acls_single)
