# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["WalkMove"]

import numpy as np
from ..compat import xrange
from .red_blue import RedBlueMove


class WalkMove(RedBlueMove):
    """
    A `Goodman & Weare (2010)
    <http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml>`_ "walk move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <http://arxiv.org/abs/1202.3665>`_.

    :param s: (optional)
        The number of helper walkers to use. By default it will use all the
        walkers in the complement.

    """
    def __init__(self, s=None, **kwargs):
        self.s = s
        super(WalkMove, self).__init__(**kwargs)

    def get_proposal(self, ens, s, c):
        Ns, Nc = len(s), len(c)
        q = np.empty((Ns, ens.ndim), dtype=np.float64)
        s0 = Nc if self.s is None else self.s
        for i in xrange(Ns):
            inds = ens.random.choice(Nc, s0, replace=False)
            cov = np.atleast_2d(np.cov(c[inds], rowvar=0))
            q[i] = ens.random.multivariate_normal(s[i], cov)
        return q, np.zeros(Ns, dtype=np.float64)
