# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["DEMove"]

import numpy as np
from ..compat import xrange
from .red_blue import RedBlueMove


class DEMove(RedBlueMove):
    """
    A `differential evolution
    <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_ proposal
    implemented following `Nelson et al. (2013)
    <http://arxiv.org/abs/1311.5229>`_.

    :param sigma:
        The standard deviation of the Gaussian used to stretch the proposal
        vector.

    :param gamma0: (optional)
        The mean stretch factor for the proposal vector. By default, it is
        :math:`2.38 / \sqrt{2\,\mathrm{ndim}}` as recommended by MAGIC and the
        two references.

    """
    def __init__(self, sigma, gamma0=None, **kwargs):
        self.sigma = sigma
        self.gamma0 = gamma0
        super(DEMove, self).__init__(**kwargs)

    def setup(self, ensemble):
        self.g0 = self.gamma0
        if self.g0 is None:
            # Fuckin' MAGIC.
            self.g0 = 2.38 / np.sqrt(2 * ensemble.ndim)

    def get_proposal(self, ens, s, c):
        Ns, Nc = len(s), len(c)
        q = np.empty((Ns, ens.ndim), dtype=np.float64)
        f = ens.random.randn(Ns)
        for i in xrange(Ns):
            inds = ens.random.choice(Nc, 2, replace=False)
            g = np.diff(c[inds], axis=0) * (1 + self.g0 * f[i])
            q[i] = s[i] + g
        return q, np.zeros(Ns, dtype=np.float64)
