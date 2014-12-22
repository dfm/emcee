# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["StretchProposal"]

import numpy as np

from ..compat import izip
from .proposal import Proposal


class StretchProposal(Proposal):

    def __init__(self, a=2.0, **kwargs):
        self.a = a
        super(StretchProposal, self).__init__(**kwargs)

    def update(self, ensemble):
        nwalkers, ndim = ensemble.nwalkers, ensemble.ndim
        ensemble.acceptance = np.zeros(nwalkers, dtype=bool)

        # Split the ensemble in half and iterate over these two halves.
        halfk = int(nwalkers / 2)
        first, second = slice(halfk), slice(halfk, nwalkers)
        for S1, S2 in [(first, second), (second, first)]:
            # Get the two halves of the ensemble.
            s = ensemble.coords[S1]
            c = ensemble.coords[S2]
            Ns = len(s)
            Nc = len(c)

            # Generate the vectors of random numbers that will produce the
            # proposal.
            zz = ((self.a - 1.) * self.random.rand(Ns) + 1) ** 2. / self.a
            factors = (ndim - 1.) * np.log(zz)
            rint = self.random.randint(Nc, size=(Ns,))

            # Calculate the proposed positions and compute the lnprobs.
            q = c[rint] - (c[rint] - s) * zz[:, None]
            new_walkers = ensemble.propose(q, S1)

            # Loop over the walkers and update them accordingly.
            for i, f, w in izip(np.arange(nwalkers)[S1], factors, new_walkers):
                lnpdiff = f + w.lnprob - ensemble.walkers[i].lnprob
                if lnpdiff > np.log(self.random.rand()):
                    ensemble.walkers[i] = w
                    ensemble.acceptance[i] = True

            # Update the ensemble with the accepted walkers.
            ensemble.update()

        return ensemble
