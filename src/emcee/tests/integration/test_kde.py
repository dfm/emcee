# -*- coding: utf-8 -*-

from emcee import moves

from .test_proposal import _test_normal, _test_uniform

__all__ = ["test_normal_kde", "test_uniform_kde", "test_nsplits_kde"]


def test_normal_kde(**kwargs):
    _test_normal(moves.KDEMove(), **kwargs)


def test_uniform_kde(**kwargs):
    _test_uniform(moves.KDEMove(), **kwargs)


def test_nsplits_kde(**kwargs):
    _test_normal(moves.KDEMove(nsplits=5), **kwargs)
