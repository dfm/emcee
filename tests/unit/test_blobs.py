# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np

from emcee import backends, EnsembleSampler

__all__ = ["test_blob_shape"]


class BlobLogProb(object):

    def __init__(self, blob_function):
        self.blob_function = blob_function

    def __call__(self, params):
        return -0.5 * np.sum(params**2), self.blob_function(params)


@pytest.mark.parametrize(
    "backend", [backends.Backend, backends.hdf.TempHDFBackend],
)
def test_blob_shape(backend):
    with backend() as be:
        np.random.seed(42)

        nblobs = 5
        model = BlobLogProb(lambda x: np.random.randn(nblobs))

        coords = np.random.randn(32, 3)
        nwalkers, ndim = coords.shape
        sampler = EnsembleSampler(nwalkers, ndim, model, backend=be)
        nsteps = 10
        sampler.run_mcmc(coords, nsteps)
        assert sampler.get_blobs().shape == (nsteps, nwalkers, nblobs)

        model = BlobLogProb(lambda x: np.random.randn())
        be.reset(nwalkers, ndim)
        sampler = EnsembleSampler(nwalkers, ndim, model, backend=be)
        sampler.run_mcmc(coords, nsteps)
        assert sampler.get_blobs().shape == (nsteps, nwalkers)

        # HDF backend doesn't support the object type
        if backend == backends.hdf.TempHDFBackend:
            return

        model = BlobLogProb(lambda x: "face")
        be.reset(nwalkers, ndim)
        sampler = EnsembleSampler(nwalkers, ndim, model, backend=be)
        sampler.run_mcmc(coords, nsteps)
        assert sampler.get_blobs().shape == (nsteps, nwalkers)

        model = BlobLogProb(lambda x: (np.random.randn(nblobs), "face"))
        be.reset(nwalkers, ndim)
        sampler = EnsembleSampler(nwalkers, ndim, model, backend=be)
        sampler.run_mcmc(coords, nsteps)
        assert sampler.get_blobs().shape == (nsteps, nwalkers, 2)
