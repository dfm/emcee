# mypy: ignore-errors

import jax
import jax.numpy as jnp
from jax.tree_util import tree_structure, tree_flatten
import numpy as np
import pytest

from emcee.ravel_util import ravel_ensemble

ensembles_and_shapes = [
    (jnp.ones((5, 3)), (5, 3)),
    (
        {"x": jnp.zeros((3, 2)), "y": (jnp.ones(3), 2 + jnp.zeros((3, 2, 4)))},
        (3, 11),
    ),
    ((jnp.zeros((3, 2)), jnp.ones((3, 4), dtype=int)), (3, 6)),
]


@pytest.mark.parametrize("ensemble,shape", ensembles_and_shapes)
def test_shape(ensemble, shape):
    flat, _ = ravel_ensemble(ensemble)
    assert flat.shape == shape


@pytest.mark.parametrize("ensemble", [e for e, _ in ensembles_and_shapes])
def test_round_trip(ensemble):
    flat, unravel = ravel_ensemble(ensemble)
    computed = jax.vmap(unravel)(flat)
    assert tree_structure(computed) == tree_structure(ensemble)
    for a, b in zip(tree_flatten(computed)[0], tree_flatten(ensemble)[0]):
        assert a.dtype == b.dtype
        np.testing.assert_allclose(a, b)
