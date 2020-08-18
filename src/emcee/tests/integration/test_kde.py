# -*- coding: utf-8 -*-

try:
    import scipy
except ImportError:
    scipy = None
import pytest

from emcee import moves

from .test_proposal import _test_normal, _test_uniform

__all__ = ["test_normal_kde", "test_uniform_kde", "test_nsplits_kde"]


@pytest.mark.skipif(scipy is None, reason="scipy is not available")
def test_normal_kde(**kwargs):
    _test_normal(moves.KDEMove(), **kwargs)


@pytest.mark.skipif(scipy is None, reason="scipy is not available")
def test_uniform_kde(**kwargs):
    _test_uniform(moves.KDEMove(), **kwargs)


@pytest.mark.skipif(scipy is None, reason="scipy is not available")
def test_nsplits_kde(**kwargs):
    _test_normal(moves.KDEMove(nsplits=5), **kwargs)
