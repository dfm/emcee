#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["GaussianProposal"]

import copy
import numpy as np

from .base import Proposal


class GaussianProposal(Proposal):

    def __init__(self, cov):
        self.cov = np.array(cov)

    def update_one(self, state):
        # Compute the ln-probability or access the cached value.
        lnprob = state.get_lnprob()

        # Generate a Gaussian proposal.
        pos = state.get_coords()
        pos = self._random.multivariate_normal(pos, self.cov)

        # # Build a propose state at this new position.
        # proposed_state = initial_state.factory(pos)

        # Compute update probability.
        newlnprob = state.lnprob(pos)
        accept_lnprob = newlnprob - lnprob
        if accept_lnprob < 0:
            accept_lnprob = np.exp(accept_lnprob) - self._random.rand()

        # Do the update.
        accept = False
        if accept_lnprob > 0:
            state.update(pos, newlnprob)
            accept = True
        return accept

    def update(self, ensemble, random=None):
        if random is None:
            self._random = np.random
        else:
            self._random = random
        [self.update_one(s) for s in ensemble]
