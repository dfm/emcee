# -*- coding: utf-8 -*-

import pickle
from itertools import product, islice
from copy import deepcopy

import numpy as np
import pytest
import packaging

from emcee import EnsembleSampler, backends, moves, walkers_independent

__all__ = ["test_shapes", "test_errors", "test_thin", "test_vectorize"]

all_backends = backends.get_test_backends()


def normal_log_prob(params):
    return -0.5 * np.sum(params ** 2)


@pytest.mark.parametrize(
    "backend, moves",
    product(
        all_backends,
        [
            None,
            moves.GaussianMove(0.5),
            [moves.StretchMove(), moves.GaussianMove(0.5)],
            [(moves.StretchMove(), 0.3), (moves.GaussianMove(0.5), 0.1)],
        ],
    ),
)
def test_shapes(backend, moves, nwalkers=32, ndim=3, nsteps=10, seed=1234):
    # Set up the random number generator.
    np.random.seed(seed)

    with backend() as be:
        # Initialize the ensemble, moves and sampler.
        coords = np.random.randn(nwalkers, ndim)
        sampler = EnsembleSampler(
            nwalkers, ndim, normal_log_prob, moves=moves, backend=be
        )

        # Run the sampler.
        sampler.run_mcmc(coords, nsteps)

        chain = sampler.get_chain()
        assert len(chain) == nsteps, "wrong number of steps"

        tau = sampler.get_autocorr_time(quiet=True)
        assert tau.shape == (ndim,)

        # Check the shapes.
        with pytest.warns(DeprecationWarning):
            assert sampler.chain.shape == (
                nwalkers,
                nsteps,
                ndim,
            ), "incorrect coordinate dimensions"
        with pytest.warns(DeprecationWarning):
            assert sampler.lnprobability.shape == (
                nwalkers,
                nsteps,
            ), "incorrect probability dimensions"
        assert sampler.get_chain().shape == (
            nsteps,
            nwalkers,
            ndim,
        ), "incorrect coordinate dimensions"
        assert sampler.get_log_prob().shape == (
            nsteps,
            nwalkers,
        ), "incorrect probability dimensions"

        assert sampler.acceptance_fraction.shape == (
            nwalkers,
        ), "incorrect acceptance fraction dimensions"

        # Check the shape of the flattened coords.
        assert sampler.get_chain(flat=True).shape == (
            nsteps * nwalkers,
            ndim,
        ), "incorrect coordinate dimensions"
        assert sampler.get_log_prob(flat=True).shape == (
            nsteps * nwalkers,
        ), "incorrect probability dimensions"


@pytest.mark.parametrize("backend", all_backends)
def test_errors(backend, nwalkers=32, ndim=3, nsteps=5, seed=1234):
    # Set up the random number generator.
    np.random.seed(seed)

    with backend() as be:
        # Initialize the ensemble, proposal, and sampler.
        coords = np.random.randn(nwalkers, ndim)
        sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob, backend=be)

        # Test for not running.
        with pytest.raises(AttributeError):
            sampler.get_chain()
        with pytest.raises(AttributeError):
            sampler.get_log_prob()

        # What about not storing the chain.
        sampler.run_mcmc(coords, nsteps, store=False)
        with pytest.raises(AttributeError):
            sampler.get_chain()

        # Now what about if we try to continue using the sampler with an
        # ensemble of a different shape.
        sampler.run_mcmc(coords, nsteps, store=False)

        coords2 = np.random.randn(nwalkers, ndim + 1)
        with pytest.raises(ValueError):
            list(sampler.run_mcmc(coords2, nsteps))

        # Ensure that a warning is logged if the inital coords don't allow
        # the chain to explore all of parameter space, and that one is not
        # if we explicitly disable it, or the initial coords can.
        with pytest.raises(ValueError):
            sampler.run_mcmc(np.ones((nwalkers, ndim)), nsteps)
        sampler.run_mcmc(
            np.ones((nwalkers, ndim)), nsteps, skip_initial_state_check=True
        )
        sampler.run_mcmc(np.random.randn(nwalkers, ndim), nsteps)


def run_sampler(
    backend,
    nwalkers=32,
    ndim=3,
    nsteps=25,
    seed=1234,
    thin=None,
    thin_by=1,
    progress=False,
    store=True,
):
    np.random.seed(seed)
    coords = np.random.randn(nwalkers, ndim)
    np.random.seed(None)
    sampler = EnsembleSampler(
        nwalkers, ndim, normal_log_prob, backend=backend, seed=seed
    )
    sampler.run_mcmc(
        coords,
        nsteps,
        thin=thin,
        thin_by=thin_by,
        progress=progress,
        store=store,
    )
    return sampler


@pytest.mark.parametrize("backend", all_backends)
def test_thin(backend):
    with backend() as be:
        with pytest.raises(ValueError):
            with pytest.warns(DeprecationWarning):
                run_sampler(be, thin=-1)
        with pytest.raises(ValueError):
            with pytest.warns(DeprecationWarning):
                run_sampler(be, thin=0.1)
        thinby = 3
        sampler1 = run_sampler(None)
        with pytest.warns(DeprecationWarning):
            sampler2 = run_sampler(be, thin=thinby)
        for k in ["get_chain", "get_log_prob"]:
            a = getattr(sampler1, k)()[thinby - 1 :: thinby]
            b = getattr(sampler2, k)()
            c = getattr(sampler1, k)(thin=thinby)
            assert np.allclose(a, b), "inconsistent {0}".format(k)
            assert np.allclose(a, c), "inconsistent {0}".format(k)


@pytest.mark.parametrize(
    "backend,progress", product(all_backends, [True, False])
)
def test_thin_by(backend, progress):
    with backend() as be:
        with pytest.raises(ValueError):
            run_sampler(be, thin_by=-1)
        with pytest.raises(ValueError):
            run_sampler(be, thin_by=0.1)
        nsteps = 25
        thinby = 3
        sampler1 = run_sampler(None, nsteps=nsteps * thinby, progress=progress)
        sampler2 = run_sampler(
            be, thin_by=thinby, progress=progress, nsteps=nsteps
        )
        for k in ["get_chain", "get_log_prob"]:
            a = getattr(sampler1, k)()[thinby - 1 :: thinby]
            b = getattr(sampler2, k)()
            c = getattr(sampler1, k)(thin=thinby)
            assert np.allclose(a, b), "inconsistent {0}".format(k)
            assert np.allclose(a, c), "inconsistent {0}".format(k)
        assert sampler1.iteration == sampler2.iteration * thinby


@pytest.mark.parametrize("backend", all_backends)
def test_restart(backend):
    with backend() as be:
        sampler = run_sampler(be, nsteps=0)
        with pytest.raises(ValueError):
            sampler.run_mcmc(None, 10)

        sampler = run_sampler(be)
        sampler.run_mcmc(None, 10)

    with backend() as be:
        sampler = run_sampler(be, store=False)
        sampler.run_mcmc(None, 10)


def test_vectorize():
    def lp_vec(p):
        return -0.5 * np.sum(p ** 2, axis=1)

    np.random.seed(42)
    nwalkers, ndim = 32, 3
    coords = np.random.randn(nwalkers, ndim)
    sampler = EnsembleSampler(nwalkers, ndim, lp_vec, vectorize=True)
    sampler.run_mcmc(coords, 10)

    assert sampler.get_chain().shape == (10, nwalkers, ndim)


@pytest.mark.parametrize("backend", all_backends)
def test_pickle(backend):
    with backend() as be:
        sampler1 = run_sampler(be)
        s = pickle.dumps(sampler1, -1)
        sampler2 = pickle.loads(s)
        for k in ["get_chain", "get_log_prob"]:
            a = getattr(sampler1, k)()
            b = getattr(sampler2, k)()
            assert np.allclose(a, b), "inconsistent {0}".format(k)


@pytest.mark.parametrize("nwalkers, ndim", [(10, 2), (20, 5)])
def test_walkers_dependent_ones(nwalkers, ndim):
    assert not walkers_independent(np.ones((nwalkers, ndim)))


@pytest.mark.parametrize("nwalkers, ndim", [(10, 11), (2, 3)])
def test_walkers_dependent_toofew(nwalkers, ndim):
    assert not walkers_independent(np.random.randn(nwalkers, ndim))


@pytest.mark.parametrize("nwalkers, ndim", [(10, 2), (20, 5)])
def test_walkers_independent_randn(nwalkers, ndim):
    assert walkers_independent(np.random.randn(nwalkers, ndim))


@pytest.mark.parametrize(
    "nwalkers, ndim, offset", [(10, 2, 1e5), (20, 5, 1e10), (30, 10, 1e14)]
)
def test_walkers_independent_randn_offset(nwalkers, ndim, offset):
    assert walkers_independent(
        np.random.randn(nwalkers, ndim) + np.ones((nwalkers, ndim)) * offset
    )


def test_walkers_dependent_big_offset():
    nwalkers, ndim = 30, 10
    offset = 10 / np.finfo(float).eps
    assert not walkers_independent(
        np.random.randn(nwalkers, ndim) + np.ones((nwalkers, ndim)) * offset
    )


def test_walkers_dependent_subtle():
    nwalkers, ndim = 30, 10
    w = np.random.randn(nwalkers, ndim)
    assert walkers_independent(w)
    # random unit vector
    p = np.random.randn(ndim)
    p /= np.sqrt(np.dot(p, p))
    # project away the direction of p
    w -= np.sum(p[None, :] * w, axis=1)[:, None] * p[None, :]
    assert not walkers_independent(w)
    # shift away from the origin
    w += p[None, :]
    assert not walkers_independent(w)


def test_walkers_almost_dependent():
    nwalkers, ndim = 30, 10
    squash = 1e-8
    w = np.random.randn(nwalkers, ndim)
    assert walkers_independent(w)
    # random unit vector
    p = np.random.randn(ndim)
    p /= np.sqrt(np.dot(p, p))
    # project away the direction of p
    proj = np.sum(p[None, :] * w, axis=1)[:, None] * p[None, :]
    w -= proj
    w += squash * proj
    assert not walkers_independent(w)


def test_walkers_independent_scaled():
    # Some of these scales will overflow if squared, hee hee
    scales = np.array([1, 1e10, 1e100, 1e200, 1e-10, 1e-100, 1e-200])
    ndim = len(scales)
    nwalkers = 5 * ndim
    w = np.random.randn(nwalkers, ndim) * scales[None, :]
    assert walkers_independent(w)


@pytest.mark.parametrize(
    "nwalkers, ndim, offset",
    [
        (10, 2, 1e5),
        (20, 5, 1e10),
        (30, 10, 1e14),
        (40, 15, 0.1 / np.finfo(np.longdouble).eps),
    ],
)
def test_walkers_independent_randn_offset_longdouble(nwalkers, ndim, offset):
    assert walkers_independent(
        np.random.randn(nwalkers, ndim)
        + np.ones((nwalkers, ndim), dtype=np.longdouble) * offset
    )

@pytest.mark.parametrize("backend", all_backends)
def test_infinite_iterations_store(backend, nwalkers=32, ndim=3):
    with backend() as be:
        coords = np.random.randn(nwalkers, ndim)
        with pytest.raises(ValueError):
            next(EnsembleSampler(nwalkers, ndim, normal_log_prob, backend=be).sample(coords, iterations=None, store=True))

@pytest.mark.parametrize("backend", all_backends)
def test_infinite_iterations(backend, nwalkers=32, ndim=3):
    with backend() as be:
        coords = np.random.randn(nwalkers, ndim)
        for state in islice(EnsembleSampler(nwalkers, ndim, normal_log_prob, backend=be).sample(coords, iterations=None, store=False), 10):
            pass

def test_sampler_seed():
    nwalkers = 32
    ndim = 3
    nsteps = 25
    np.random.seed(456)
    coords = np.random.randn(nwalkers, ndim)
    sampler1 = EnsembleSampler(nwalkers, ndim, normal_log_prob, seed=1234)
    sampler2 = EnsembleSampler(nwalkers, ndim, normal_log_prob, seed=2)
    sampler3 = EnsembleSampler(nwalkers, ndim, normal_log_prob, seed=1234)
    sampler4 = EnsembleSampler(
        nwalkers, ndim, normal_log_prob, seed=deepcopy(sampler1._random)
    )
    for sampler in (sampler1, sampler2, sampler3, sampler4):
        sampler.run_mcmc(coords, nsteps)
    for k in ["get_chain", "get_log_prob"]:
        attr1 = getattr(sampler1, k)()
        attr2 = getattr(sampler2, k)()
        attr3 = getattr(sampler3, k)()
        attr4 = getattr(sampler4, k)()
        assert not np.allclose(attr1, attr2), "inconsistent {0}".format(k)
        np.testing.assert_allclose(attr1, attr3, err_msg="inconsistent {0}".format(k))
        np.testing.assert_allclose(attr1, attr4, err_msg="inconsistent {0}".format(k))


def test_sampler_bad_seed():
    nwalkers = 32
    ndim = 3
    with pytest.raises(TypeError, match="seed must be"):
        EnsembleSampler(nwalkers, ndim, normal_log_prob, seed="bad_seed")

@pytest.mark.skipif(
    packaging.version.parse(np.__version__) < packaging.version.parse("1.17.0"),
    reason="requires numpy 1.17.0 or higher",
)
def test_sampler_generator():
    nwalkers = 32
    ndim = 3
    nsteps = 5
    np.random.seed(456)
    coords = np.random.randn(nwalkers, ndim)
    seed1 = np.random.default_rng(1)
    sampler1 = EnsembleSampler(nwalkers, ndim, normal_log_prob, seed=seed1)
    sampler1.run_mcmc(coords, nsteps)
    seed2 = np.random.default_rng(1)
    sampler2 = EnsembleSampler(nwalkers, ndim, normal_log_prob, seed=seed2)
    sampler2.run_mcmc(coords, nsteps)
    np.testing.assert_allclose(sampler1.get_chain(), sampler2.get_chain())
    np.testing.assert_allclose(sampler1.get_log_prob(), sampler2.get_log_prob())
