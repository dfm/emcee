# -*- coding: utf-8 -*-

from emcee import moves

from .test_proposal import _test_normal, _test_uniform

__all__ = ["test_normal_de_snooker", "test_uniform_de_snooker"]


def test_normal_de_snooker(**kwargs):
    kwargs["nsteps"] = 4000
    _test_normal(moves.DESnookerMove(), **kwargs)


def test_uniform_de_snooker(**kwargs):
    _test_uniform(moves.DESnookerMove(), **kwargs)
