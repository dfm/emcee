# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_normal_gaussian", "test_uniform_gaussian"]

import numpy as np
from ... import moves
from .test_proposal import _test_normal, _test_uniform


def test_normal_gaussian(**kwargs):
    _test_normal(moves.GaussianMove(0.5), **kwargs)


def test_normal_gaussian_nd(**kwargs):
    ndim = 3

    # Isotropic.
    _test_normal(moves.GaussianMove(0.5), ndim=ndim, **kwargs)

    # Axis-aligned.
    _test_normal(moves.GaussianMove(0.5 * np.ones(ndim)), ndim=ndim, **kwargs)
    try:
        _test_normal(moves.GaussianMove(0.5 * np.ones(ndim-1)), ndim=ndim,
                     **kwargs)
    except ValueError:
        pass
    else:
        assert 0, "should raise a ValueError"

    # Full matrix.
    _test_normal(moves.GaussianMove(np.diag(0.5 * np.ones(ndim))), ndim=ndim,
                 **kwargs)
    try:
        _test_normal(moves.GaussianMove(np.diag(0.5 * np.ones(ndim-1))),
                     ndim=ndim, **kwargs)
    except ValueError:
        pass
    else:
        assert 0, "should raise a ValueError"


def test_uniform_gaussian(**kwargs):
    _test_uniform(moves.GaussianMove(0.5), **kwargs)
