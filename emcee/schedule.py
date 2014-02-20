#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Schedule"]

import numpy as np


class _lnprob_wrapper(object):

    def __init__(self, lnprobfn):
        self.lnprobfn = lnprobfn

    def __call__(self, positions, blobs):
        return np.array(map(self.lnprobfn, positions))


class Schedule(object):

    def __init__(self, lnprobfn, proposals):
        self.lnprobfn = _lnprob_wrapper(lnprobfn)
        self.proposals = proposals

    def sample(self, positions, lnprobs=None, blobs=None):
        positions = np.array(positions)
        if lnprobs is None:
            lnprobs = self.lnprobfn(positions, blobs)
        else:
            lnprobs = np.array(lnprobs)

        while True:
            for prop in self.proposals:
                accept = prop.update(self.lnprobfn, positions, lnprobs, blobs)
                yield accept
