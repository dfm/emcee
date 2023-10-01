# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np

from .red_blue import RedBlueMove

__all__ = ["DEMove"]


class DEMove(RedBlueMove):
    r"""A proposal using differential evolution.

    This `Differential evolution proposal
    <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_ is
    implemented following `Nelson et al. (2013)
    <https://doi.org/10.1088/0067-0049/210/1/11>`_.

    Args:
        sigma (float): The standard deviation of the Gaussian used to stretch
            the proposal vector.
        gamma0 (Optional[float]): The mean stretch factor for the proposal
            vector. By default, it is :math:`2.38 / \sqrt{2\,\mathrm{ndim}}`
            as recommended by the two references.

    """

    def __init__(self, sigma=1.0e-5, gamma0=None, **kwargs):
        self.sigma = sigma
        self.gamma0 = gamma0
        super().__init__(**kwargs)

    def setup(self, coords):
        self.g0 = self.gamma0
        if self.g0 is None:
            # Pure MAGIC:
            ndim = coords.shape[1]
            self.g0 = 2.38 / np.sqrt(2 * ndim)

    def get_proposal(self, s, c, random):
        c = np.concatenate(c, axis=0)
        ns, ndim = s.shape
        nc = c.shape[0]

        # Get the pair indices
        pairs = _get_nondiagonal_pairs(nc)

        # Sample from the pairs
        indices = random.choice(pairs.shape[0], size=ns, replace=True)
        pairs = pairs[indices]

        # Compute diff vectors
        diffs = np.diff(c[pairs], axis=1).squeeze(axis=1)  # (ns, ndim)

        # Sample a gamma value for each walker following Nelson et al. (2013)
        gamma = self.g0 * (1 + self.sigma * random.randn(ns, 1))  # (ns, 1)

        # In this way, sigma is the standard deviation of the distribution of gamma,
        # instead of the standard deviation of the distribution of the proposal as proposed by Ter Braak (2006).
        # Otherwise, sigma should be tuned for each dimension, which confronts the idea of affine-invariance.

        q = s + gamma * diffs

        return q, np.zeros(ns, dtype=np.float64)


@lru_cache(maxsize=1)
def _get_nondiagonal_pairs(n: int) -> np.ndarray:
    """Get the indices of a square matrix with size n, excluding the diagonal."""
    rows, cols = np.tril_indices(n, -1)  # -1 to exclude diagonal

    # Combine rows-cols and cols-rows pairs
    pairs = np.column_stack(
        [np.concatenate([rows, cols]), np.concatenate([cols, rows])]
    )

    return pairs
