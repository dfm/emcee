"""
This module is based on the implementation of ``ravel_pytree`` in
``jax.flatten_util``, but it adds support for the leading dimension encountered
in an ensemble.
"""

__all__ = ["ravel_ensemble"]

import warnings
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import dtypes
from jax._src.util import safe_zip
from jax.tree_util import tree_flatten, tree_unflatten

from emcee.types import Array, Params, UnravelFn

zip = safe_zip


def ravel_ensemble(coords: Params) -> Tuple[Array, UnravelFn]:
    leaves, treedef = tree_flatten(coords)
    flat, unravel_inner = _ravel_inner(leaves)
    unravel_one = lambda flat: tree_unflatten(treedef, unravel_inner(flat))
    return flat, unravel_one


def _ravel_inner(lst: List[Array]) -> Tuple[Array, UnravelFn]:
    if not lst:
        return jnp.array([], jnp.float32), lambda _: []
    from_dtypes = [dtypes.dtype(l) for l in lst]
    to_dtype = dtypes.result_type(*from_dtypes)
    shapes = [jnp.shape(x)[1:] for x in lst]
    indices = np.cumsum([int(np.prod(s)) for s in shapes])

    if all(dt == to_dtype for dt in from_dtypes):
        del from_dtypes, to_dtype

        def unravel(arr: Array) -> Params:
            chunks = jnp.split(arr, indices[:-1])
            return [
                chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)
            ]

        ravel = lambda arg: jnp.concatenate([jnp.ravel(e) for e in arg])
        raveled = jax.vmap(ravel)(lst)
        return raveled, unravel

    else:

        def unravel(arr: Array) -> Params:
            arr_dtype = dtypes.dtype(arr)
            if arr_dtype != to_dtype:
                raise TypeError(
                    f"unravel function given array of dtype {arr_dtype}, "
                    f"but expected dtype {to_dtype}"
                )
            chunks = jnp.split(arr, indices[:-1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return [
                    lax.convert_element_type(chunk.reshape(shape), dtype)
                    for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)
                ]

        ravel = lambda arg: jnp.concatenate(
            [jnp.ravel(lax.convert_element_type(e, to_dtype)) for e in arg]
        )
        raveled = jax.vmap(ravel)(lst)
        return raveled, unravel
