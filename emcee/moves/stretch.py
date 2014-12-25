# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["StretchMove"]

import numpy as np
from .red_blue import RedBlueMove


class StretchMove(RedBlueMove):
    """
    A `Goodman & Weare (2010)
    <http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <http://arxiv.org/abs/1202.3665>`_.

    :param a: (optional)
        The stretch scale parameter. (default: ``2.0``)

    """
    def __init__(self, a=2.0, **kwargs):
        self.a = a
        super(StretchMove, self).__init__(**kwargs)

    def get_proposal(self, ens, s, c):
        Ns, Nc = len(s), len(c)
        zz = ((self.a - 1.) * ens.random.rand(Ns) + 1) ** 2. / self.a
        factors = (ens.ndim - 1.) * np.log(zz)
        rint = ens.random.randint(Nc, size=(Ns,))
        return c[rint] - (c[rint] - s) * zz[:, None], factors
