# -*- coding: utf-8 -*-

import os
from itertools import product
from os.path import join

import numpy as np
import pytest

from emcee import EnsembleSampler, State, backends
from emcee.backends.hdf import does_hdf5_support_longdouble

try:
    import h5py
except ImportError:
    h5py = None

__all__ = ["test_backend", "test_reload"]

all_backends = backends.get_test_backends()
other_backends = all_backends[1:]
dtypes = [None, [("log_prior", float), ("mean", int)]]


def normal_log_prob(params):
    return -0.5 * np.sum(params ** 2)


def normal_log_prob_blobs(params):
    return normal_log_prob(params), 0.1, int(5)


def run_sampler(
    backend,
    nwalkers=32,
    ndim=3,
    nsteps=25,
    seed=1234,
    thin_by=1,
    dtype=None,
    blobs=True,
    lp=None,
):
    if lp is None:
        lp = normal_log_prob_blobs if blobs else normal_log_prob
    if seed is not None:
        np.random.seed(seed)
    coords = np.random.randn(nwalkers, ndim)
    sampler = EnsembleSampler(
        nwalkers, ndim, lp, backend=backend, blobs_dtype=dtype
    )
    sampler.run_mcmc(coords, nsteps, thin_by=thin_by)
    return sampler


def _custom_allclose(a, b):
    if a.dtype.fields is None:
        assert np.allclose(a, b)
    else:
        for n in a.dtype.names:
            assert np.allclose(a[n], b[n])


@pytest.mark.skipif(h5py is None, reason="HDF5 not available")
def test_uninit(tmpdir):
    fn = str(tmpdir.join("EMCEE_TEST_FILE_DO_NOT_USE.h5"))
    if os.path.exists(fn):
        os.remove(fn)

    with backends.HDFBackend(fn) as be:
        run_sampler(be)

    assert os.path.exists(fn)
    os.remove(fn)


@pytest.mark.parametrize("backend", all_backends)
def test_uninit_errors(backend):
    with backend() as be:
        with pytest.raises(AttributeError):
            be.get_last_sample()

        for k in ["chain", "log_prob", "blobs"]:
            with pytest.raises(AttributeError):
                getattr(be, "get_" + k)()


@pytest.mark.parametrize("backend", all_backends)
def test_blob_usage_errors(backend):
    with backend() as be:
        run_sampler(be, blobs=True)
        with pytest.raises(ValueError):
            run_sampler(be, blobs=False)

    with backend() as be:
        run_sampler(be, blobs=False)
        with pytest.raises(ValueError):
            run_sampler(be, blobs=True)


@pytest.mark.parametrize(
    "backend,dtype,blobs", product(other_backends, dtypes, [True, False])
)
def test_backend(backend, dtype, blobs):
    # Run a sampler with the default backend.
    sampler1 = run_sampler(backends.Backend(), dtype=dtype, blobs=blobs)

    with backend() as be:
        sampler2 = run_sampler(be, dtype=dtype, blobs=blobs)

        values = ["chain", "log_prob"]
        if blobs:
            values += ["blobs"]
        else:
            assert sampler1.get_blobs() is None
            assert sampler2.get_blobs() is None

        # Check all of the components.
        for k in values:
            a = getattr(sampler1, "get_" + k)()
            b = getattr(sampler2, "get_" + k)()
            _custom_allclose(a, b)

        last1 = sampler1.get_last_sample()
        last2 = sampler2.get_last_sample()
        assert np.allclose(last1.coords, last2.coords)
        assert np.allclose(last1.log_prob, last2.log_prob)
        assert all(
            np.allclose(l1, l2)
            for l1, l2 in zip(last1.random_state[1:], last2.random_state[1:])
        )
        if blobs:
            _custom_allclose(last1.blobs, last2.blobs)
        else:
            assert last1.blobs is None and last2.blobs is None

        a = sampler1.acceptance_fraction
        b = sampler2.acceptance_fraction
        assert np.allclose(a, b), "inconsistent acceptance fraction"


@pytest.mark.parametrize("backend,dtype", product(other_backends, dtypes))
def test_reload(backend, dtype):
    with backend() as backend1:
        run_sampler(backend1, dtype=dtype)

        # Test the state
        state = backend1.random_state
        np.random.set_state(state)

        # Load the file using a new backend object.
        backend2 = backends.HDFBackend(
            backend1.filename, backend1.name, read_only=True
        )

        with pytest.raises(RuntimeError):
            backend2.reset(32, 3)

        assert state[0] == backend2.random_state[0]
        assert all(
            np.allclose(a, b)
            for a, b in zip(state[1:], backend2.random_state[1:])
        )

        # Check all of the components.
        for k in ["chain", "log_prob", "blobs"]:
            a = backend1.get_value(k)
            b = backend2.get_value(k)
            _custom_allclose(a, b)

        last1 = backend1.get_last_sample()
        last2 = backend2.get_last_sample()
        assert np.allclose(last1.coords, last2.coords)
        assert np.allclose(last1.log_prob, last2.log_prob)
        assert all(
            np.allclose(l1, l2)
            for l1, l2 in zip(last1.random_state[1:], last2.random_state[1:])
        )
        _custom_allclose(last1.blobs, last2.blobs)

        a = backend1.accepted
        b = backend2.accepted
        assert np.allclose(a, b), "inconsistent accepted"


@pytest.mark.parametrize("backend,dtype", product(other_backends, dtypes))
def test_restart(backend, dtype):
    # Run a sampler with the default backend.
    b = backends.Backend()
    run_sampler(b, dtype=dtype)
    sampler1 = run_sampler(b, seed=None, dtype=dtype)

    with backend() as be:
        run_sampler(be, dtype=dtype)
        sampler2 = run_sampler(be, seed=None, dtype=dtype)

        # Check all of the components.
        for k in ["chain", "log_prob", "blobs"]:
            a = getattr(sampler1, "get_" + k)()
            b = getattr(sampler2, "get_" + k)()
            _custom_allclose(a, b)

        last1 = sampler1.get_last_sample()
        last2 = sampler2.get_last_sample()
        assert np.allclose(last1.coords, last2.coords)
        assert np.allclose(last1.log_prob, last2.log_prob)
        assert all(
            np.allclose(l1, l2)
            for l1, l2 in zip(last1.random_state[1:], last2.random_state[1:])
        )
        _custom_allclose(last1.blobs, last2.blobs)

        a = sampler1.acceptance_fraction
        b = sampler2.acceptance_fraction
        assert np.allclose(a, b), "inconsistent acceptance fraction"


@pytest.mark.skipif(h5py is None, reason="HDF5 not available")
def test_multi_hdf5():
    with backends.TempHDFBackend() as backend1:
        run_sampler(backend1)

        backend2 = backends.HDFBackend(backend1.filename, name="mcmc2")
        run_sampler(backend2)
        chain2 = backend2.get_chain()

        with h5py.File(backend1.filename, "r") as f:
            assert set(f.keys()) == {backend1.name, "mcmc2"}

        backend1.reset(10, 2)
        assert np.allclose(backend2.get_chain(), chain2)
        with pytest.raises(AttributeError):
            backend1.get_chain()


@pytest.mark.parametrize("backend", all_backends)
def test_longdouble_preserved(backend):
    if (
        issubclass(backend, backends.TempHDFBackend)
        and not does_hdf5_support_longdouble()
    ):
        pytest.xfail("HDF5 does not support long double on this platform")
    nwalkers = 10
    ndim = 2
    nsteps = 5
    with backend(dtype=np.longdouble) as b:
        b.reset(nwalkers, ndim)
        b.grow(nsteps, None)
        for i in range(nsteps):
            coords = np.zeros((nwalkers, ndim), dtype=np.longdouble)
            coords += i + 1
            coords += np.arange(nwalkers)[:, None]
            coords[:, 1] += coords[:, 0] * 2 * np.finfo(np.longdouble).eps
            assert not np.any(coords[:, 1] == coords[:, 0])
            lp = 1 + np.arange(nwalkers) * np.finfo(np.longdouble).eps
            state = State(coords, log_prob=lp, random_state=())
            b.save_step(state, np.ones((nwalkers,), dtype=bool))
            s = b.get_last_sample()
            # check s has adequate precision and equals state
            assert s.coords.dtype == np.longdouble
            assert not np.any(s.coords[:, 1] == s.coords[:, 0])
            assert np.all(s.coords == coords)

            assert s.log_prob.dtype == np.longdouble
            assert np.all(s.log_prob == lp)


@pytest.mark.skipif(h5py is None, reason="HDF5 not available")
def test_hdf5_compression():
    with backends.TempHDFBackend(compression="gzip") as b:
        run_sampler(b, blobs=True)
        # re-open and read
        b.get_chain()
        b.get_blobs()
        b.get_log_prob()
        b.accepted
