# -*- coding: utf-8 -*-
import pytest
import packaging
import numpy as np

from emcee import moves

from .test_proposal import _test_normal, _test_uniform


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="default"),
        pytest.param(
            {"nwalkers": 10, "nsteps": 10, "generator": True},
            marks=pytest.mark.skipif(
                packaging.version.parse(np.__version__) < packaging.version.parse("1.17.0"),
                reason="requires numpy 1.17.0 or higher",
            ),
            id="Generator"
        )
    ]
)
class TestDE:
    def test_normal_de(self, kwargs):
        _test_normal(moves.DEMove(), **kwargs)

    def test_normal_de_no_gamma(self, kwargs):
        _test_normal(moves.DEMove(gamma0=1.0), **kwargs)

    def test_uniform_de(self, kwargs):
        _test_uniform(moves.DEMove(), **kwargs)
