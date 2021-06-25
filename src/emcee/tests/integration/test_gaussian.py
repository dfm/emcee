# -*- coding: utf-8 -*-

from itertools import product

import numpy as np
import pytest

from emcee import moves

from .test_proposal import _test_normal, _test_uniform

__all__ = [
    "test_normal_gaussian",
    "test_uniform_gaussian",
    "test_normal_gaussian_nd",
]


@pytest.mark.parametrize("mode,factor", product(["vector"], [None, 2.0, 5.0]))
def test_normal_gaussian(mode, factor, **kwargs):
    _test_normal(moves.GaussianMove(0.5, mode=mode, factor=factor), **kwargs)


@pytest.mark.parametrize(
    "mode,factor", product(["vector", "random", "sequential"], [None, 2.0])
)
def test_normal_gaussian_nd(mode, factor, **kwargs):
    ndim = 3
    kwargs["nsteps"] = 8000

    # Isotropic.
    _test_normal(
        moves.GaussianMove(0.5, factor=factor, mode=mode), ndim=ndim, **kwargs
    )

    # Axis-aligned.
    _test_normal(
        moves.GaussianMove(0.5 * np.ones(ndim), factor=factor, mode=mode),
        ndim=ndim,
        **kwargs,
    )
    with pytest.raises(ValueError):
        _test_normal(
            moves.GaussianMove(
                0.5 * np.ones(ndim - 1), factor=factor, mode=mode
            ),
            ndim=ndim,
            **kwargs,
        )

    # Full matrix.
    if mode == "vector":
        _test_normal(
            moves.GaussianMove(
                np.diag(0.5 * np.ones(ndim)), factor=factor, mode=mode
            ),
            ndim=ndim,
            **kwargs,
        )
        with pytest.raises(ValueError):
            _test_normal(
                moves.GaussianMove(np.diag(0.5 * np.ones(ndim - 1))),
                ndim=ndim,
                **kwargs,
            )
    else:
        with pytest.raises(ValueError):
            _test_normal(
                moves.GaussianMove(
                    np.diag(0.5 * np.ones(ndim)), factor=factor, mode=mode
                ),
                ndim=ndim,
                **kwargs,
            )


@pytest.mark.parametrize("mode,factor", product(["vector"], [None, 2.0, 5.0]))
def test_uniform_gaussian(mode, factor, **kwargs):
    _test_uniform(moves.GaussianMove(0.5, factor=factor, mode=mode), **kwargs)
