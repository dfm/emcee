#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["GaussianProposal"]

import numpy as np
from itertools import izip, imap

from .base import Proposal


class GaussianProposal(Proposal):

    def __init__(self, cov, random=None):
        self.cov = np.array(cov)
        if random is None:
            self._random = np.random
        else:
            self._random = random

    def update(self, lnprobfn, positions, lnprobs, blobs):
        nwalkers, ndim = positions.shape
        q = self._random.multivariate_normal(np.zeros(ndim), self.cov,
                                             size=nwalkers)
        q += positions

        newlnprobs = lnprobfn(q, blobs)
        acceptlnprobs = newlnprobs - lnprobs
        m = acceptlnprobs < 0
        acceptlnprobs[m] = np.exp(acceptlnprobs[m]) - self._random.rand(sum(m))

        accept = acceptlnprobs > 0
        positions[accept] = q[accept, :]
        lnprobs[accept] = newlnprobs[accept]

        return accept
