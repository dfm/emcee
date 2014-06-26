# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["StretchProposal"]

import numpy as np

from .base import Proposal


class StretchProposal(Proposal):

    def __init__(self, a=2.0, **kwargs):
        self.a = a
        super(StretchProposal, self).__init__(**kwargs)

    def update(self, ens_lnprob_fn, coords_in, lnprior_in, lnlike_in,
               coords_out, lnprior_out, lnlike_out):
        # Parse the dimensions of the input coordinates.
        coords_in = np.atleast_2d(coords_in)
        nens, ndim = coords_in.shape

        # Save the initial values into a working array.
        coords_out[:] = np.array(coords_in)
        lnprior_out[:] = np.array(lnprior_in)
        lnlike_out[:] = np.array(lnlike_in)
        acc = np.zeros(nens, dtype=bool)

        # Split the ensemble in half and iterate over these two halves.
        halfk = int(nens / 2)
        first, second = slice(halfk), slice(halfk, nens)
        for S1, S2 in [(first, second), (second, first)]:
            # Get the two halves of the ensemble.
            s = coords_out[S1]
            c = coords_out[S2]
            Ns = len(s)
            Nc = len(c)

            # Generate the vectors of random numbers that will produce the
            # proposal.
            zz = ((self.a - 1.) * self.random.rand(Ns) + 1) ** 2. / self.a
            rint = self.random.randint(Nc, size=(Ns,))

            # Calculate the proposed positions.
            q = c[rint] - zz[:, np.newaxis] * (c[rint] - s)

            # Compute the lnprior and lnlikelihood at the new positions.
            newlnprior, newlnlike = ens_lnprob_fn(q)

            # Decide whether or not the proposals should be accepted.
            lnpdiff = ((ndim - 1.) * np.log(zz) + (newlnprior + newlnlike)
                       - (lnprior_out[S1] + lnlike_out[S1]))
            accept = (lnpdiff > np.log(self.random.rand(len(lnpdiff))))

            # Update the positions.
            coords_out[S1][accept] = q[accept]
            lnprior_out[S1][accept] = newlnprior[accept]
            lnlike_out[S1][accept] = newlnlike[accept]
            acc[S1] += accept

        return acc
