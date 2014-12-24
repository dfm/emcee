# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["NormalWalker", "UniformWalker"]

import numpy as np
from .. import BaseWalker


class NormalWalker(BaseWalker):

    def __init__(self, coords, ivar):
        self.ivar = ivar
        super(NormalWalker, self).__init__(coords, ivar)

    def lnpriorfn(self, p):
        return 0.0

    def lnlikefn(self, p):
        return -0.5 * np.sum(p ** 2 * self.ivar)


class UniformWalker(BaseWalker):

    def lnpriorfn(self, p):
        return 0.0 if np.all((-1 < p) * (p < 1)) else -np.inf

    def lnlikefn(self, p):
        return 0.0
