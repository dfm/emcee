# -*- coding: utf-8 -*-

from __future__ import division, print_function

from multiprocessing import Pool
import pytest
from emcee import moves
from .test_proposal import _test_normal

__all__ = ["test_normal_nuts"]


@pytest.mark.parametrize("pool", [True, False])
def test_normal_nuts(pool, **kwargs):
    if pool:
        kwargs["pool"] = Pool()
    kwargs["check_acceptance"] = False
    kwargs["nsteps"] = 100
    _test_normal(moves.NUTSMove(), **kwargs)
    if pool:
        kwargs["pool"].close()
