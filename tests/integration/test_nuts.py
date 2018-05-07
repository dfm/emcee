# -*- coding: utf-8 -*-

from __future__ import division, print_function

from multiprocessing import Pool
import pytest
import numpy as np

from emcee import moves
from emcee.moves.nuts import (
    IdentityMetric, IsotropicMetric, DiagonalMetric, DenseMetric
)

from .test_proposal import _test_normal

__all__ = ["test_normal_nuts"]


@pytest.mark.parametrize("metric", [None, IdentityMetric(3),
                                    IsotropicMetric(3),
                                    DiagonalMetric(np.ones(3)),
                                    DenseMetric(np.eye(3))])
@pytest.mark.parametrize("pool", [True, False])
@pytest.mark.parametrize("tune", [True, False])
def test_normal_nuts(pool, metric, tune, **kwargs):
    if pool:
        kwargs["pool"] = Pool()
    if tune:
        move = moves.NUTSMove(ntune=300)
    else:
        move = moves.NUTSMove()
    kwargs["ndim"] = 3
    kwargs["check_acceptance"] = False
    kwargs["nsteps"] = 100
    _test_normal(move, **kwargs)
    if pool:
        kwargs["pool"].close()
