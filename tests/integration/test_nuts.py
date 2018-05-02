# -*- coding: utf-8 -*-

from __future__ import division, print_function

from emcee import moves
from .test_proposal import _test_normal

__all__ = ["test_normal_nuts"]


def test_normal_nuts(**kwargs):
    kwargs["check_acceptance"] = False
    kwargs["nsteps"] = 100
    _test_normal(moves.NUTSMove(), **kwargs)
