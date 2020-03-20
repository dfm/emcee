# -*- coding: utf-8 -*-
import pytest
import packaging
import numpy as np

from emcee import moves

from .test_proposal import _test_normal, _test_uniform


@pytest.mark.parametrize(
    "generator",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                packaging.version.parse(np.__version__) < packaging.version.parse("1.17.0"),
                reason="requires numpy 1.17.0 or higher",
            )
        )
    ]
)
class TestDE:
    def test_normal_de(self, generator):
        _test_normal(moves.DEMove(), generator=generator)

    def test_normal_de_no_gamma(self, kwargs):
        _test_normal(moves.DEMove(gamma0=1.0), generator=generator)

    def test_uniform_de(self, kwargs):
        _test_uniform(moves.DEMove(), generator=generator)
