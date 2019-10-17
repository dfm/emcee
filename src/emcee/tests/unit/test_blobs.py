# -*- coding: utf-8 -*-

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
        (True, 5, lambda x: np.random.randn(5)),
        (True, 0, lambda x: np.random.randn()),
        (False, 2, lambda x: (1.0, np.random.randn(3))),
        (False, 0, lambda x: "face"),
        (False, 2, lambda x: (np.random.randn(5), "face")),
    ],
)
def test_blob_shape(backend, blob_spec):
    # HDF backends don't support the object type
    if backend in (backends.TempHDFBackend,) and not blob_spec[0]:
        return

    with backend() as be:
        np.random.seed(42)

        model = BlobLogProb(blob_spec[2])
        coords = np.random.randn(32, 3)
        nwalkers, ndim = coords.shape

        sampler = EnsembleSampler(nwalkers, ndim, model, backend=be)
        nsteps = 10

        sampler.run_mcmc(coords, nsteps)

        shape = [nsteps, nwalkers]
        if blob_spec[1] > 0:
            shape += [blob_spec[1]]

        assert sampler.get_blobs().shape == tuple(shape)
