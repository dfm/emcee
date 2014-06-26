# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Proposal"]

import numpy as np


class Proposal(object):

    def __init__(self, random=None):
        if random is None:
            self.random = np.random
        else:
            self.random = random

    def update(self, ens_lnprob_fn, coords_in, lnlike_in, lnprior_in,
               coords_out, lnlike_out, lnprior_out):
        raise NotImplementedError("Subclasses must implement the update "
                                  "method")
