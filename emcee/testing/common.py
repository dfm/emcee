# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["NormalWalker", "UniformWalker"]

import os
import numpy as np
from tempfile import NamedTemporaryFile

from .. import BaseWalker, backends


class NormalWalker(BaseWalker):

    def __init__(self, ivar):
        self.ivar = ivar

    def lnpriorfn(self, p):
        return 0.0

    def lnlikefn(self, p):
        return -0.5 * np.sum(p ** 2 * self.ivar)


class UniformWalker(BaseWalker):

    def lnpriorfn(self, p):
        return 0.0 if np.all((-1 < p) * (p < 1)) else -np.inf

    def lnlikefn(self, p):
        return 0.0


class TempHDFBackend(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        f = NamedTemporaryFile("w", delete=False)
        f.close()
        self.filename = f.name
        return backends.HDFBackend(f.name, "test", **(self.kwargs))

    def __exit__(self, exception_type, exception_value, traceback):
        os.remove(self.filename)
