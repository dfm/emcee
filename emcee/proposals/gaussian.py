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

    def update_one(self, initial_state):
        # Compute the ln-probability or access the cached value.
        lnprob = initial_state.get_lnprob()

        # Generate a Gaussian proposal.
        pos = initial_state.get_coords()
        pos = self._random.multivariate_normal(pos, self.cov)

        # Build a propose state at this new position.
        proposed_state = initial_state.factory(pos)

        # Compute update probability.
        accept_lnprob = proposed_state.get_lnprob() - lnprob
        if accept_lnprob < 0:
            accept_lnprob = np.exp(accept_lnprob) - self._random.rand()

        # Do the update.
        if accept_lnprob > 0:
            return proposed_state, True
        return initial_state, False

    def update(self, initial_ensemble, random=None):
        if random is None:
            self._random = np.random
        else:
            self._random = random
        return zip(*map(self.update_one, initial_ensemble))
