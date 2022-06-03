from functools import wraps
from typing import Any, Callable, Tuple

from jax import random
import jax.numpy as jnp

from emcee.types import (
    InitFn,
    StepFn,
    Array,
    State,
    WrappedLogProbFn,
    Stats,
    MoveFn,
)

ProposalFn = Callable[[random.KeyArray, Array, Array], Tuple[Array, Array]]


def red_blue(
    build_proposal: Callable[..., Tuple[InitFn, ProposalFn]],
) -> MoveFn:
    @wraps(build_proposal)
    def move_impl(
        log_prob_fn: WrappedLogProbFn, *args: Any, **kwargs: Any
    ) -> Tuple[InitFn, StepFn]:
        init, proposal = build_proposal(*args, **kwargs)

        def step(
            state: State,
            random_key: random.KeyArray,
            ensemble: Array,
            log_prob: Array,
        ) -> Tuple[Stats, Array, Array]:
            key1, key2 = random.split(random_key)
            nwalkers, _ = ensemble.shape
            mid = nwalkers // 2

            a = ensemble[:mid]
            b = ensemble[mid:]

            def half_step(
                key: random.KeyArray, s: Array, c: Array, lp: Array
            ) -> Tuple[Array, Array, Array]:
                key1, key2 = random.split(key)
                q, f = proposal(key1, s, c)
                nlp = log_prob_fn(q)
                diff = f + nlp - lp
                accept = jnp.exp(diff) > random.uniform(key2, shape=lp.shape)
                return (
                    accept,
                    jnp.where(accept[:, None], q, s),
                    jnp.where(accept, nlp, lp),
                )

            acc1, a, lp1 = half_step(key1, a, b, log_prob[:mid])
            acc2, b, lp2 = half_step(key2, b, a, log_prob[mid:])

            return (
                {"accept": jnp.concatenate((acc1, acc2))},
                jnp.concatenate((a, b)),
                jnp.concatenate((lp1, lp2)),
            )

        return init, step

    return move_impl


@red_blue
def stretch(*, a: float = 2.0) -> Tuple[InitFn, ProposalFn]:
    def proposal(
        key: random.KeyArray, s: Array, c: Array
    ) -> Tuple[Array, Array]:
        ns, ndim = s.shape
        key1, key2 = random.split(key)
        u = random.uniform(key1, shape=(ns,))
        z = jnp.square((a - 1) * u + 1) / a
        c = random.choice(key2, c, shape=(ns,))
        q = c - (c - s) * z[..., None]
        return q, (ndim - 1) * jnp.log(z)

    return lambda _: None, proposal
