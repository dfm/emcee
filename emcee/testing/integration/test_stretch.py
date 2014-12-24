# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_normal_stretch", "test_uniform_stretch"]

import numpy as np
from scipy import stats

from ...compat import xrange
from ... import proposals, Ensemble

from ..common import NormalWalker, UniformWalker


def test_normal_stretch(nwalkers=32, nsteps=2000, seed=1234):
    # Set up the random number generator.
    rnd = np.random.RandomState()
    rnd.seed(seed)

    # Initialize the ensemble and proposal.
    coords = rnd.randn(nwalkers, 1)
    ensemble = Ensemble(NormalWalker, coords, 1.0, random=rnd)
    proposal = proposals.StretchProposal()

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

    # Check the resulting chain using a K-S test and compare to the mean and
    # standard deviation.
    samps = chain.flatten()
    np.random.shuffle(samps)
    ks, _ = stats.kstest(samps, "norm")
    mu, sig = np.mean(samps), np.std(samps)
    assert ks < 0.05, "The K-S test failed"
    assert np.abs(mu) < 0.05, "Incorrect mean"
    assert np.abs(sig - 1) < 0.05, "Incorrect standard deviation"


def test_uniform_stretch(nwalkers=32, nsteps=2000, seed=1234):
    # Set up the random number generator.
    rnd = np.random.RandomState()
    rnd.seed(seed)

    # Initialize the ensemble and proposal.
    coords = 2*rnd.rand(nwalkers, 1) - 1
    ensemble = Ensemble(UniformWalker, coords, random=rnd)
    proposal = proposals.StretchProposal()

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
