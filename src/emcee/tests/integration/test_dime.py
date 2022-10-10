# -*- coding: utf-8 -*-

from emcee import moves

from .test_proposal import _test_normal, _test_uniform

__all__ = ["test_normal_dime", "test_uniform_de"]


def test_normal_dime(**kwargs):
    _test_normal(moves.DIMEMove(), **kwargs)


def test_uniform_dime(**kwargs):
    _test_uniform(moves.DIMEMove(), **kwargs)
