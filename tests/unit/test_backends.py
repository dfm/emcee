# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import pytest

from emcee import backends, EnsembleSampler

__all__ = ["test_backend", "test_reload"]

all_backends = backends.get_test_backends()[1:]


def normal_log_prob(params):
    return -0.5 * np.sum(params**2)


def run_sampler(backend, nwalkers=32, ndim=3, nsteps=25, seed=1234, thin_by=1):
    np.random.seed(seed)
    coords = np.random.randn(nwalkers, ndim)
    sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob,
                              backend=backend)
    sampler.run_mcmc(coords, nsteps, thin_by=thin_by)
    return sampler


@pytest.mark.parametrize("backend", all_backends)
def test_backend(backend):
    # Run a sampler with the default backend.
    sampler1 = run_sampler(backends.Backend())

    with backend() as be:
        sampler2 = run_sampler(be)

        # Check all of the components.
        for k in ["chain", "log_prob"]:
            a = getattr(sampler1, "get_" + k)()
            b = getattr(sampler2, "get_" + k)()
            assert np.allclose(a, b), "inconsistent {0}".format(k)

        a = sampler1.acceptance_fraction
        b = sampler2.acceptance_fraction
        assert np.allclose(a, b), "inconsistent acceptance fraction"


@pytest.mark.parametrize("backend", all_backends)
def test_reload(backend):
    with backend() as backend1:
        run_sampler(backend1)

        # Test the state
        state = backend1.random_state
        np.random.set_state(state)

        # Load the file using a new backend object.
        if backend == backends.TempHDFBackend:
            backend2 = backends.HDFBackend(backend1.filename, backend1.name)
        elif backend == backends.TempFITSBackend:
            backend2 = backends.FITSBackend(backend1.filename,
                                            backend1.pickle_filename)
        else:
            assert False

        assert state[0] == backend2.random_state[0]
        assert all(np.allclose(a, b)
                   for a, b in zip(state[1:], backend2.random_state[1:]))

        # Check all of the components.
        for k in ["chain", "log_prob"]:
            a = backend1.get_value(k)
            b = backend2.get_value(k)
            assert np.allclose(a, b), "inconsistent {0}".format(k)

        a = backend1.accepted
        b = backend2.accepted
        assert np.allclose(a, b), "inconsistent accepted"
