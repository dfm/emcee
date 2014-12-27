# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["_test_normal", "_test_uniform"]

import numpy as np
from scipy import stats

from ... import Ensemble
from ...compat import xrange
from ..common import NormalWalker, UniformWalker


def _test_normal(proposal, ndim=1, nwalkers=32, nsteps=2000, seed=1234):
    # Set up the random number generator.
    rnd = np.random.RandomState()
    rnd.seed(seed)

    # Initialize the ensemble and proposal.
    coords = rnd.randn(nwalkers, ndim)
    ensemble = Ensemble(NormalWalker, coords, 1.0, random=rnd)

    # Run the chain.
    chain = np.empty((nsteps, nwalkers, ndim))
    acc = np.zeros(nwalkers, dtype=int)
    for i in xrange(len(chain)):
        proposal.update(ensemble)
        chain[i] = ensemble.coords
        acc += ensemble.acceptance

    # Check the acceptance fraction.
    acc = acc / nsteps
    assert np.all((acc < 0.9) * (acc > 0.1)), \
        "Invalid acceptance fraction\n{0}".format(acc)

    # Check the resulting chain using a K-S test and compare to the mean and
    # standard deviation.
    samps = chain.flatten()
    mu, sig = np.mean(samps, axis=0), np.std(samps, axis=0)
    assert np.all(np.abs(mu) < 0.05), "Incorrect mean"
    assert np.all(np.abs(sig - 1) < 0.05), "Incorrect standard deviation"

    if ndim == 1:
        ks, _ = stats.kstest(samps, "norm")
        assert ks < 0.05, "The K-S test failed"


def _test_uniform(proposal, nwalkers=32, nsteps=2000, seed=1234):
    # Set up the random number generator.
    rnd = np.random.RandomState()
    rnd.seed(seed)

    # Initialize the ensemble and proposal.
    coords = 2*rnd.rand(nwalkers, 1) - 1
    ensemble = Ensemble(UniformWalker, coords, random=rnd)

    # Run the chain.
    chain = np.empty((nsteps, nwalkers, 1))
    acc = np.zeros(nwalkers, dtype=int)
    for i in xrange(len(chain)):
        proposal.update(ensemble)
        chain[i] = ensemble.coords
        acc += ensemble.acceptance

    # Check the acceptance fraction.
    acc = acc / nsteps
    assert np.all((acc < 0.9) * (acc > 0.1)), \
        "Invalid acceptance fraction\n{0}".format(acc)

    # Check that the resulting chain "fails" the K-S test.
    samps = chain.flatten()
    np.random.shuffle(samps)
    ks, _ = stats.kstest(samps, "norm")
    assert ks > 0.1, "The K-S test failed"
