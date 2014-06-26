# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["StretchProposal"]

import numpy as np

from .base import Proposal


class StretchProposal(Proposal):

    def __init__(self, a=2.0, **kwargs):
        self.a = a
        super(StretchProposal, self).__init__(**kwargs)

    def update(self, state_in, state_out):
        nens, ndim = state_in.shape
        state_out.update(state_in)
        acc = np.zeros(nens, dtype=bool)

        # Split the ensemble in half and iterate over these two halves.
        halfk = int(nens / 2)
        first, second = slice(halfk), slice(halfk, nens)
        for S1, S2 in [(first, second), (second, first)]:
            # Get the two halves of the ensemble.
            s = state_out[S1]
            c = state_out[S2]
            Ns = len(s)
            Nc = len(c)

            # Generate the vectors of random numbers that will produce the
            # proposal.
            zz = ((self.a - 1.) * self.random.rand(Ns) + 1) ** 2. / self.a
            rint = self.random.randint(Nc, size=(Ns,))

            # Calculate the proposed positions.
            q = c[rint] - (c[rint] - s) * zz[:, None]

            # Compute the lnprior and lnlikelihood at the new positions.
            q.compute_lnprob()

            # Decide whether or not the proposals should be accepted.
            lnpdiff = (ndim - 1.) * np.log(zz) + q.lnprob - s.lnprob
            accept = (lnpdiff > np.log(self.random.rand(len(lnpdiff))))

            # Update the positions.
            s.update(q, accept)
            acc[S1] += accept

        return acc
