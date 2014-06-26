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

    def update(self, lnprob_fn, state_in, state_out):
        raise NotImplementedError("Subclasses must implement the update "
                                  "method")
