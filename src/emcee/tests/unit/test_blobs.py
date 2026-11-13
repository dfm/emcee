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
        return -0.5 * np.sum(params**2), self.blob_function(params)


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


class PlaceholderBlobLogProb:
    def __init__(self, placeholder):
        self.placeholder = placeholder

    def __call__(self, params):
        if params[0] < 0:
            # Blobs returned with a log probability of -inf are not tracked
            return -np.inf, self.placeholder
        return -0.5 * np.sum(params**2), params[0]


def test_compute_log_prob_preserves_array_blobs_with_none_placeholder():
    def log_prob(params):
        if params[0] < 0:
            return -np.inf, None
        return -params[0] ** 2, np.array([params[0], 2 * params[0]])

    coords = np.array([[-1.0], [0.75]])
    sampler = EnsembleSampler(2, 1, log_prob)
    log_probs, blobs = sampler.compute_log_prob(coords)
    np.testing.assert_array_equal(log_probs, [-np.inf, -0.5625])
    assert blobs.dtype == object
    assert blobs[0] is None
    np.testing.assert_array_equal(blobs[1], [0.75, 1.5])

    # An explicitly requested dtype must still be enforced.
    sampler = EnsembleSampler(2, 1, log_prob, blobs_dtype=float)
    with pytest.raises(ValueError):
        sampler.compute_log_prob(coords)


@pytest.mark.parametrize("backend", backends.get_test_backends())
@pytest.mark.parametrize("placeholder", [0, False, None])
@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_blob_dtype_ignores_placeholders(backend, placeholder):
    with backend() as be:
        np.random.seed(42)

        model = PlaceholderBlobLogProb(placeholder)
        coords = np.abs(np.random.randn(32, 3)) + 0.1
        coords[0, 0] = -1.0
        nwalkers, ndim = coords.shape

        sampler = EnsembleSampler(nwalkers, ndim, model, backend=be)
        sampler.run_mcmc(coords, 20)

        blobs = sampler.get_blobs()
        assert blobs.dtype == np.float64
        tracked = np.isfinite(sampler.get_log_prob())
        x = sampler.get_chain()[..., 0]
        np.testing.assert_allclose(blobs[tracked], x[tracked])
