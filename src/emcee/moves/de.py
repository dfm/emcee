# -*- coding: utf-8 -*-

import numpy as np

from .red_blue import RedBlueMove

__all__ = ["DEMove"]


class DEMove(RedBlueMove):
    r"""A proposal using differential evolution.

    This `Differential evolution proposal
    <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_ is
    implemented following `Nelson et al. (2013)
    <https://arxiv.org/abs/1311.5229>`_.

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
        kwargs["nsplits"] = 3
        super(DEMove, self).__init__(**kwargs)

    def setup(self, coords):
        self.g0 = self.gamma0
        if self.g0 is None:
            # Pure MAGIC:
            ndim = coords.shape[1]
            self.g0 = 2.38 / np.sqrt(2 * ndim)

    def get_proposal(self, s, c, random):
        Ns = len(s)
        Nc = list(map(len, c))
        ndim = s.shape[1]
        q = np.empty((Ns, ndim), dtype=np.float64)
        f = self.sigma * random.randn(Ns)
        for i in range(Ns):
            w = np.array([c[j][random.randint(Nc[j])] for j in range(2)])
            random.shuffle(w)
            g = np.diff(w, axis=0) * self.g0 + f[i]
            q[i] = s[i] + g
        return q, np.zeros(Ns, dtype=np.float64)
