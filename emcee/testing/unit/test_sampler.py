# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_schedule", "test_shapes", "test_errors", "test_walkers"]

import numpy as np

from ... import moves, backends, Sampler, Ensemble
from ..common import NormalWalker, TempHDFBackend


def test_schedule():
    # The default schedule should be a single stretch move.
    s = Sampler()
    assert len(s.schedule) == 1

    # A single move.
    s = Sampler(moves.GaussianMove(0.5))
    assert len(s.schedule) == 1

    # A list of moves.
    s = Sampler([moves.StretchMove(), moves.GaussianMove(0.5)])
    assert len(s.schedule) == 2


def test_shapes():
    run_shapes(backends.DefaultBackend(store_walkers=True))
    with TempHDFBackend(store_walkers=True) as backend:
        run_shapes(backend)


def test_walkers():
    run_walkers(backends.DefaultBackend(store_walkers=True))
    with TempHDFBackend(store_walkers=True) as backend:
        run_walkers(backend)


def run_shapes(backend, nwalkers=32, ndim=3, nsteps=5, seed=1234):
    # Set up the random number generator.
    rnd = np.random.RandomState()
    rnd.seed(seed)

    # Initialize the ensemble, proposal, and sampler.
    coords = rnd.randn(nwalkers, ndim)
    ensemble = Ensemble(NormalWalker, coords, 1.0, random=rnd)
    sampler = Sampler(backend=backend)

    # Run the sampler.
    ensembles = list(sampler.sample(ensemble, nsteps))
    assert len(ensembles) == nsteps, "wrong number of steps"

    # Check the shapes.
    assert sampler.coords.shape == (nsteps, nwalkers, ndim), \
        "incorrect coordinate dimensions"

    assert sampler.lnprior.shape == (nsteps, nwalkers), \
        "incorrect prior dimensions"
    assert sampler.lnlike.shape == (nsteps, nwalkers), \
        "incorrect likelihood dimensions"
    assert sampler.lnprob.shape == (nsteps, nwalkers), \
        "incorrect probability dimensions"

    assert sampler.acceptance_fraction.shape == (nwalkers,), \
        "incorrect acceptance fraction dimensions"

    assert len(sampler.walkers) == nsteps, \
        "incorrect walker list dimensions"
    assert len(sampler.walkers[0]) == nwalkers, \
        "incorrect walker row dimensions"

    # This should work (even though it's dumb).
    sampler.reset()
    for i, e in enumerate(sampler.sample(ensemble, store=True)):
        if i >= nsteps - 1:
            break
    assert sampler.coords.shape == (nsteps, nwalkers, ndim), \
        "incorrect coordinate dimensions"
    assert sampler.lnprior.shape == (nsteps, nwalkers), \
        "incorrect prior dimensions"
    assert sampler.lnlike.shape == (nsteps, nwalkers), \
        "incorrect likelihood dimensions"
    assert sampler.lnprob.shape == (nsteps, nwalkers), \
        "incorrect probability dimensions"
    assert sampler.acceptance_fraction.shape == (nwalkers,), \
        "incorrect acceptance fraction dimensions"


def run_walkers(backend, nwalkers=32, ndim=3, nsteps=5, seed=1234):
    # Set up the random number generator.
    rnd = np.random.RandomState()
    rnd.seed(seed)

    # Initialize the ensemble, proposal, and sampler.
    coords = rnd.randn(nwalkers, ndim)
    ensemble = Ensemble(NormalWalker, coords, 1.0, random=rnd)
    sampler = Sampler(backend=backend)

    # Run the sampler.
    list(sampler.sample(ensemble, nsteps))

    for i, row in enumerate(sampler.walkers):
        for j, w in enumerate(row):
            assert np.allclose(sampler.coords[i, j], w.coords), \
                "invalid walker coordinates"


def test_errors(nwalkers=32, ndim=3, nsteps=5, seed=1234):
    # Set up the random number generator.
    rnd = np.random.RandomState()
    rnd.seed(seed)

    # Initialize the ensemble, proposal, and sampler.
    coords = rnd.randn(nwalkers, ndim)
    ensemble = Ensemble(NormalWalker, coords, 1.0, random=rnd)

    # Test for saving the walker list.
    sampler = Sampler()
    list(sampler.sample(ensemble, nsteps))
    try:
        sampler.walkers
    except AttributeError:
        pass
    else:
        assert 0, "shouldn't save walkers"

    # Test for not running.
    sampler = Sampler()
    try:
        print(sampler.coords)
    except AttributeError:
        pass
    else:
        assert 0, "should raise AttributeError"
    try:
        print(sampler.lnprob)
    except AttributeError:
        pass
    else:
        assert 0, "should raise AttributeError"

    # What about not storing the chain.
    list(sampler.sample(ensemble, nsteps, store=False))
    try:
        sampler.coords
    except AttributeError:
        pass
    else:
        assert 0, "should raise AttributeError"

    # Now what about if we try to continue using the sampler with an ensemble
    # of a different shape.
    list(sampler.sample(ensemble, nsteps))

    coords2 = rnd.randn(nwalkers, ndim+1)
    ensemble2 = Ensemble(NormalWalker, coords2, 1.0, random=rnd)
    try:
        list(sampler.sample(ensemble2, nsteps))
    except ValueError:
        pass
    else:
        assert 0, "should raise ValueError"

    # Iterating without an end state shouldn't save the chain.
    for i, e in enumerate(sampler.sample(ensemble)):
        if i >= nsteps:
            break
    try:
        sampler.coords
    except AttributeError:
        pass
    else:
        assert 0, "should raise AttributeError"
