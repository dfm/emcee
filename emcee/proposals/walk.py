# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["WalkProposal"]

import numpy as np
from ..compat import xrange, imap, izip


class WalkProposal(object):
    """
    A `Goodman & Weare (2010)
    <http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml>`_ "walk move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <http://arxiv.org/abs/1202.3665>`_.

    :param live_dangerously: (optional)
        By default, an update will fail with a ``RuntimeError`` if the number
        of walkers is smaller than twice the dimension of the problem because
        the walkers would then be stuck on a low dimensional subspace. This
        can be avoided by switching between the stretch move and, for example,
        a Metropolis-Hastings step. If you want to do this and suppress the
        error, set ``live_dangerously = True``. Thanks goes (once again) to
        @dstndstn for this wonderful terminology.

    """
    def __init__(self, s=None, live_dangerously=False):
        self.s = s
        self.live_dangerously = live_dangerously

    def update(self, ensemble):
        """
        Execute a single walk move starting from the given
        :class:`Ensemble` and updating it in-place.

        :param ensemble:
            The starting :class:`Ensemble`.

        :return ensemble:
            The same ensemble updated in-place.

        """
        nwalkers, ndim = ensemble.nwalkers, ensemble.ndim
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError("It is unadvisable to use the stretch move "
                               "with fewer walkers than twice the number of "
                               "dimensions.")
        ensemble.acceptance[:] = False

        # Split the ensemble in half and iterate over these two halves.
        halfk = int(nwalkers / 2)
        first, second = slice(halfk), slice(halfk, nwalkers)
        for S1, S2 in [(first, second), (second, first)]:
            # Get the two halves of the ensemble.
            s = ensemble.coords[S1]
            c = ensemble.coords[S2]
            Ns = len(s)
            Nc = len(c)

            # Compute the proposed coordinates.
            q = np.empty((Ns, ndim), dtype=np.float64)
            s0 = Nc if self.s is None else self.s
            for i in xrange(Ns):
                inds = ensemble.random.choice(Nc, s0, replace=False)
                cov = np.atleast_2d(np.cov(c[inds], rowvar=0))
                q[i] = ensemble.random.multivariate_normal(s[i], cov)

            # Compute the lnprobs.
            new_walkers = ensemble.propose(q, S1)

            # Loop over the walkers and update them accordingly.
            for i, w in izip(np.arange(nwalkers)[S1], new_walkers):
                lnpdiff = w.lnprob - ensemble.walkers[i].lnprob
                if lnpdiff > np.log(ensemble.random.rand()):
                    ensemble.walkers[i] = w
                    ensemble.acceptance[i] = True

            # Update the ensemble with the accepted walkers.
            ensemble.update()

        return ensemble
