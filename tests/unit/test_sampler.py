# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np
from emcee import moves, EnsembleSampler

__all__ = ["test_shapes", "test_errors", "test_thin"]


def normal_log_prob(params):
    return -0.5 * np.sum(params**2)


@pytest.mark.parametrize("moves", [
    None,
    moves.GaussianMove(0.5),
    [moves.StretchMove(), moves.GaussianMove(0.5)],
    [(moves.StretchMove(), 0.3), (moves.GaussianMove(0.5), 0.1)],
])
def test_shapes(moves, nwalkers=32, ndim=3, nsteps=100, seed=1234):
    # Set up the random number generator.
    np.random.seed(seed)

    # Initialize the ensemble, moves and sampler.
    coords = np.random.randn(nwalkers, ndim)
    sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob, moves=moves)

    # Run the sampler.
    sampler.run_mcmc(coords, nsteps)
    chain = sampler.get_chain()
    assert len(chain) == nsteps, "wrong number of steps"

    tau = sampler.get_autocorr_time(quiet=True)
    assert tau.shape == (ndim,)

    # Check the shapes.
    assert sampler.chain.shape == (nsteps, nwalkers, ndim), \
        "incorrect coordinate dimensions"
    assert sampler.log_prob.shape == (nsteps, nwalkers), \
        "incorrect probability dimensions"

    assert sampler.acceptance_fraction.shape == (nwalkers,), \
        "incorrect acceptance fraction dimensions"

    # Check the shape of the flattened coords.
    assert sampler.get_chain(flat=True).shape == \
        (nsteps * nwalkers, ndim), "incorrect coordinate dimensions"
    assert sampler.get_log_prob(flat=True).shape == \
        (nsteps*nwalkers,), "incorrect probability dimensions"

    # assert np.allclose(sampler.current_coords, sampler.coords[-1])


def test_errors(nwalkers=32, ndim=3, nsteps=5, seed=1234):
    # Set up the random number generator.
    np.random.seed(seed)

    # Initialize the ensemble, proposal, and sampler.
    coords = np.random.randn(nwalkers, ndim)
    sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob)

    # Test for not running.
    with pytest.raises(AttributeError):
        sampler.chain
    with pytest.raises(AttributeError):
        sampler.log_prob

    # What about not storing the chain.
    sampler.run_mcmc(coords, nsteps, store=False)
    with pytest.raises(AttributeError):
        sampler.chain

    # Now what about if we try to continue using the sampler with an ensemble
    # of a different shape.
    sampler.run_mcmc(coords, nsteps, store=False)

    coords2 = np.random.randn(nwalkers, ndim+1)
    with pytest.raises(ValueError):
        list(sampler.run_mcmc(coords2, nsteps))


def run_sampler(nwalkers=32, ndim=3, nsteps=25, seed=1234, thin=1):
    np.random.seed(seed)
    coords = np.random.randn(nwalkers, ndim)
    sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob)
    sampler.run_mcmc(coords, nsteps, thin=thin)
    return sampler


def test_thin():
    thinby = 3
    sampler1 = run_sampler()
    sampler2 = run_sampler(thin=thinby)
    for k in ["chain", "log_prob"]:
        a = getattr(sampler1, k)[thinby-1::thinby]
        b = getattr(sampler2, k)
        assert np.allclose(a, b), "inconsistent {0}".format(k)


def test_restart():
    sampler = run_sampler()
    sampler.run_mcmc(None, 10)
