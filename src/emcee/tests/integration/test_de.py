# -*- coding: utf-8 -*-
import pytest
from numpy.random import default_rng

from emcee import moves

from .test_proposal import _test_normal, _test_uniform


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"nwalkers": 10, "nsteps": 10, "generator": default_rng(1)}
    ]
)
class TestDE:
    def test_normal_de(self, kwargs):
        _test_normal(moves.DEMove(), **kwargs)

    def test_normal_de_no_gamma(self, kwargs):
        _test_normal(moves.DEMove(gamma0=1.0), **kwargs)

    def test_uniform_de(self, kwargs):
        _test_uniform(moves.DEMove(), **kwargs)
