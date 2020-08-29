# -*- coding: utf-8 -*-

import warnings

import numpy as np
import pytest

from emcee import EnsembleSampler, backends

__all__ = ["test_blob_shape"]


class BlobLogProb(object):
    def __init__(self, blob_function):
        self.blob_function = blob_function

    def __call__(self, params):
        return -0.5 * np.sum(params ** 2), self.blob_function(params)


@pytest.mark.parametrize("backend", backends.get_test_backends())
@pytest.mark.parametrize(
    "blob_spec",
    [
        (True, False, 5, lambda x: np.random.randn(5)),
        (True, False, (5, 3), lambda x: np.random.randn(5, 3)),
        (True, False, (5, 3), lambda x: np.random.randn(1, 5, 1, 3, 1)),
        (True, False, 0, lambda x: np.random.randn()),
        (False, True, 2, lambda x: (1.0, np.random.randn(3))),
        (False, False, 0, lambda x: "face"),
        (False, False, 0, lambda x: object()),
        (False, False, 2, lambda x: ("face", "surface")),
        (False, True, 2, lambda x: (np.random.randn(5), "face")),
    ],
)
def test_blob_shape(backend, blob_spec):
    # HDF backends don't support the object type
    hdf_able, ragged, blob_shape, func = blob_spec
    if backend in (backends.TempHDFBackend,) and not hdf_able:
        return

    with backend() as be:
        np.random.seed(42)

        model = BlobLogProb(func)
        coords = np.random.randn(32, 3)
        nwalkers, ndim = coords.shape

        sampler = EnsembleSampler(nwalkers, ndim, model, backend=be)
        nsteps = 10

        if ragged:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                sampler.run_mcmc(coords, nsteps)
        else:
            sampler.run_mcmc(coords, nsteps)

        shape = [nsteps, nwalkers]
        if isinstance(blob_shape, tuple):
            shape += blob_shape
        elif blob_shape > 0:
            shape += [blob_shape]

        assert sampler.get_blobs().shape == tuple(shape)
        if not hdf_able:
            assert sampler.get_blobs().dtype == np.dtype("object")


class VariableLogProb:
    def __init__(self):
        self.i = 3

    def __call__(self, *args):
        return 0, np.zeros(self.i)


@pytest.mark.parametrize("backend", backends.get_test_backends())
def test_blob_mismatch(backend):
    with backend() as be:
        np.random.seed(42)

        model = VariableLogProb()
        coords = np.random.randn(32, 3)
        nwalkers, ndim = coords.shape

        sampler = EnsembleSampler(nwalkers, ndim, model, backend=be)

        model.i += 1
        # We don't save blobs from the initial points
        # so blob shapes are taken from the first round of moves
        sampler.run_mcmc(coords, 1)

        model.i += 1
        with pytest.raises(ValueError):
            sampler.run_mcmc(coords, 1)
