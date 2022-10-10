# -*- coding: utf-8 -*-

try:
    import scipy
except ImportError:
    scipy = None
import pytest

from emcee import moves

from .test_proposal import _test_normal, _test_uniform

__all__ = ["test_normal_dime", "test_uniform_de"]


@pytest.mark.skipif(scipy is None, reason="scipy is not available")
def test_normal_dime(**kwargs):
    _test_normal(moves.DIMEMove(), **kwargs)


@pytest.mark.skipif(scipy is None, reason="scipy is not available")
def test_uniform_dime(**kwargs):
    _test_uniform(moves.DIMEMove(), **kwargs)
