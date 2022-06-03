from typing import Any, Optional, Tuple, Dict
from functools import wraps

import jax
from jax import random
from emcee.moves import stretch
from emcee.types import LogProbFn, MoveFn, Stats, Walker, Array
from emcee.ravel_util import ravel_ensemble


def sample(
    random_key: random.KeyArray,
    log_prob_fn: LogProbFn,
    ensemble: Walker,
    num_steps: int,
    move: Optional[MoveFn] = None,
    *,
    log_prob_args: Tuple[Any, ...] = (),
    log_prob_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Walker, Stats]:
    log_prob_kwargs = {} if log_prob_kwargs is None else log_prob_kwargs

    @jax.vmap
    @wraps(log_prob_fn)
    def wrapped_log_prob_fn(x: Array) -> Array:
        assert log_prob_kwargs is not None
        x = unravel(x)
        return log_prob_fn(x, *log_prob_args, **log_prob_kwargs)

    move = stretch if move is None else move
    log_prob = ensemble.log_probability
    ensemble, unravel = ravel_ensemble(ensemble.coords)
    init, step = move(wrapped_log_prob_fn)
    state = init(ensemble)

    def wrapped_step(
        carry: Tuple[Array, Array], key: random.KeyArray
    ) -> Tuple[Tuple[Array, Array], Tuple[Stats, Array, Array]]:
        ensemble, log_prob = carry
        stats, ensemble, log_prob = step(state, key, ensemble, log_prob)
        return (ensemble, log_prob), (stats, ensemble, log_prob)

    carry = (ensemble, log_prob)
    _, (stats, ensemble, log_prob) = jax.lax.scan(
        wrapped_step, carry, random.split(random_key, num_steps)
    )
    return (
        Walker(
            coords=jax.vmap(jax.vmap(unravel))(ensemble),
            log_probability=log_prob,
        ),
        stats,
    )
