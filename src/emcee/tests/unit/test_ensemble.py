"""
Unit tests of some functionality in ensemble.py when the parameters are named
"""
import string
from unittest import TestCase

import numpy as np
import pytest

from emcee.ensemble import EnsembleSampler, ndarray_to_list_of_dicts


class TestNP2ListOfDicts(TestCase):
    def test_ndarray_to_list_of_dicts(self):
        # Try different numbers of keys
        for n_keys in [1, 2, 10, 26]:
            keys = list(string.ascii_lowercase[:n_keys])
            key_set = set(keys)
            key_dict = {key: i for i, key in enumerate(keys)}
            # Try different number of walker/procs
            for N in [1, 2, 3, 10, 100]:
                x = np.random.rand(N, n_keys)

                LOD = ndarray_to_list_of_dicts(x, key_dict)
                assert len(LOD) == N, "need 1 dict per row"
                for i, dct in enumerate(LOD):
                    assert dct.keys() == key_set, "keys are missing"
                    for j, key in enumerate(keys):
                        assert dct[key] == x[i, j], f"wrong value at {(i, j)}"


class TestNamedParameters(TestCase):
    """
    Test that a keyword-based log-probability function instead of
    a positional.
    """

    # Keyword based lnpdf
    def lnpdf(self, pars) -> np.float64:
        mean = pars["mean"]
        var = pars["var"]
        if var <= 0:
            return -np.inf
        return (
            -0.5 * ((mean - self.x) ** 2 / var + np.log(2 * np.pi * var)).sum()
        )

    def lnpdf_mixture(self, pars) -> np.float64:
        mean1 = pars["mean1"]
        var1 = pars["var1"]
        mean2 = pars["mean2"]
        var2 = pars["var2"]
        if var1 <= 0 or var2 <= 0:
            return -np.inf
        return (
            -0.5
            * (
                (mean1 - self.x) ** 2 / var1
                + np.log(2 * np.pi * var1)
                + (mean2 - self.x - 3) ** 2 / var2
                + np.log(2 * np.pi * var2)
            ).sum()
        )

    def lnpdf_mixture_grouped(self, pars) -> np.float64:
        mean1, mean2 = pars["means"]
        var1, var2 = pars["vars"]
        const = pars["constant"]
        if var1 <= 0 or var2 <= 0:
            return -np.inf
        return (
            -0.5
            * (
                (mean1 - self.x) ** 2 / var1
                + np.log(2 * np.pi * var1)
                + (mean2 - self.x - 3) ** 2 / var2
                + np.log(2 * np.pi * var2)
            ).sum()
            + const
        )

    def setUp(self):
        # Draw some data from a unit Gaussian
        self.x = np.random.randn(100)
        self.names = ["mean", "var"]

    def test_named_parameters(self):
        sampler = EnsembleSampler(
            nwalkers=10,
            ndim=len(self.names),
            log_prob_fn=self.lnpdf,
            parameter_names=self.names,
        )
        assert sampler.params_are_named
        assert list(sampler.parameter_names.keys()) == self.names

    def test_asserts(self):
        # ndim name mismatch
        with pytest.raises(AssertionError):
            _ = EnsembleSampler(
                nwalkers=10,
                ndim=len(self.names) - 1,
                log_prob_fn=self.lnpdf,
                parameter_names=self.names,
            )

        # duplicate names
        with pytest.raises(AssertionError):
            _ = EnsembleSampler(
                nwalkers=10,
                ndim=3,
                log_prob_fn=self.lnpdf,
                parameter_names=["a", "b", "a"],
            )

        # vectorize turned on
        with pytest.raises(AssertionError):
            _ = EnsembleSampler(
                nwalkers=10,
                ndim=len(self.names),
                log_prob_fn=self.lnpdf,
                parameter_names=self.names,
                vectorize=True,
            )

    def test_compute_log_prob(self):
        # Try different numbers of walkers
        for N in [4, 8, 10]:
            sampler = EnsembleSampler(
                nwalkers=N,
                ndim=len(self.names),
                log_prob_fn=self.lnpdf,
                parameter_names=self.names,
            )
            coords = np.random.rand(N, len(self.names))
            lnps, _ = sampler.compute_log_prob(coords)
            assert len(lnps) == N
            assert lnps.dtype == np.float64

    def test_compute_log_prob_mixture(self):
        names = ["mean1", "var1", "mean2", "var2"]
        # Try different numbers of walkers
        for N in [8, 10, 20]:
            sampler = EnsembleSampler(
                nwalkers=N,
                ndim=len(names),
                log_prob_fn=self.lnpdf_mixture,
                parameter_names=names,
            )
            coords = np.random.rand(N, len(names))
            lnps, _ = sampler.compute_log_prob(coords)
            assert len(lnps) == N
            assert lnps.dtype == np.float64

    def test_compute_log_prob_mixture_grouped(self):
        names = {"means": [0, 1], "vars": [2, 3], "constant": 4}
        # Try different numbers of walkers
        for N in [8, 10, 20]:
            sampler = EnsembleSampler(
                nwalkers=N,
                ndim=5,
                log_prob_fn=self.lnpdf_mixture_grouped,
                parameter_names=names,
            )
            coords = np.random.rand(N, 5)
            lnps, _ = sampler.compute_log_prob(coords)
            assert len(lnps) == N
            assert lnps.dtype == np.float64

    def test_run_mcmc(self):
        # Sort of an integration test
        n_walkers = 4
        sampler = EnsembleSampler(
            nwalkers=n_walkers,
            ndim=len(self.names),
            log_prob_fn=self.lnpdf,
            parameter_names=self.names,
        )
        guess = np.random.rand(n_walkers, len(self.names))
        n_steps = 50
        results = sampler.run_mcmc(guess, n_steps)
        assert results.coords.shape == (n_walkers, len(self.names))
        chain = sampler.chain
        assert chain.shape == (n_walkers, n_steps, len(self.names))
