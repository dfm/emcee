import numpy as np
import pytest

from emcee import EnsembleSampler, autocorr, backends


@pytest.mark.parametrize("backend", backends.get_test_backends())
@pytest.mark.parametrize("thin", [1, 5])
def test_autocorr_error_tau_in_original_steps(backend, thin):
    with backend() as be:
        coords = np.arange(-4, 4, dtype=float)[:, None]
        sampler = EnsembleSampler(8, 1, lambda x: -np.dot(x, x), backend=be)
        sampler.random_state = np.random.RandomState(117).get_state()
        sampler.run_mcmc(coords, 100)
        expected = thin * autocorr.integrated_time(
            sampler.get_chain(thin=thin, discard=10), tol=0
        )
        with pytest.raises(autocorr.AutocorrError) as exc:
            sampler.get_autocorr_time(thin=thin, discard=10, tol=1000)
        np.testing.assert_array_equal(exc.value.tau, expected)
        np.testing.assert_array_equal(
            sampler.get_autocorr_time(
                thin=thin, discard=10, tol=1000, quiet=True
            ),
            expected,
        )
