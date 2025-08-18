# -*- coding: utf-8 -*-

import numpy as np

from .red_blue import RedBlueMove

__all__ = ["StretchMove"]


class StretchMove(RedBlueMove):
    """
    A `Goodman & Weare (2010)
    <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <https://arxiv.org/abs/1202.3665>`_.

    :param a: (optional)
        The stretch scale parameter. (default: ``2.0``)

    """

    def __init__(self, a=2.0, **kwargs):
        self.a = a
        super(StretchMove, self).__init__(**kwargs)

    def get_proposal(self, s, c, random):
        c = np.concatenate(c, axis=0)
        Ns, Nc = len(s), len(c)
        ndim = s.shape[1]
        zz = ((self.a - 1.0) * random.rand(Ns) + 1) ** 2.0 / self.a
        factors = (ndim - 1.0) * np.log(zz)
        rint = random.randint(Nc, size=(Ns,))
        # deterministic matching of c to s
        # they should be the same size +- 1.
        if Nc == Ns:
            cc = c
        elif Nc == Ns + 1:  # s is one shorter
            cc = c[:Ns]
        elif Nc == Ns - 1:  # s is one longer, reuse one
            # RedBlueMove's permutation means the first element is not special
            cc = np.append(c, c[0])
        else:
            # old resampling behaviour; this should not occur
            rint = random.randint(Nc, size=(Ns,))
            cc = c[rint]
        return proposal = cc - (cc - s) * zz[:, None], factors
