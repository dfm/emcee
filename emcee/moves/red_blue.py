# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["RedBlueMove"]

import numpy as np
from ..compat import izip


class RedBlueMove(object):
    """
    An abstract red-blue ensemble move with parallelization as described in
    `Foreman-Mackey et al. (2013) <http://arxiv.org/abs/1202.3665>`_.

    :param live_dangerously: (optional)
        By default, an update will fail with a ``RuntimeError`` if the number
        of walkers is smaller than twice the dimension of the problem because
        the walkers would then be stuck on a low dimensional subspace. This
        can be avoided by switching between the stretch move and, for example,
        a Metropolis-Hastings step. If you want to do this and suppress the
        error, set ``live_dangerously = True``. Thanks goes (once again) to
        @dstndstn for this wonderful terminology.

    """
    def __init__(self, live_dangerously=False):
        self.live_dangerously = live_dangerously

    def setup(self, ensemble):
        pass

    def get_proposal(self, ensemble, sample, complement):
        raise NotImplementedError("The proposal must be implemented by "
                                  "subclasses")

    def finalize(self, ensemble):
        pass

    def update(self, ensemble):
        """
        Execute a move starting from the given :class:`Ensemble` and updating
        it in-place.

        :param ensemble:
            The starting :class:`Ensemble`.

        :return ensemble:
            The same ensemble updated in-place.

        """
        # Check that the dimensions are compatible.
        nwalkers, ndim = ensemble.nwalkers, ensemble.ndim
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError("It is unadvisable to use the stretch move "
                               "with fewer walkers than twice the number of "
                               "dimensions.")
        ensemble.acceptance[:] = False

        # Run any move-specific setup.
        self.setup(ensemble)

        # Split the ensemble in half and iterate over these two halves.
        halfk = int(nwalkers / 2)
        first, second = slice(halfk), slice(halfk, nwalkers)
        for S1, S2 in [(first, second), (second, first)]:
            # Get the two halves of the ensemble.
            s = ensemble.coords[S1]
            c = ensemble.coords[S2]

            # Get the move-specific proposal.
            q, factors = self.get_proposal(ensemble, s, c)

            # Compute the lnprobs of the proposed position.
            new_walkers = ensemble.propose(q, S1)

            # Loop over the walkers and update them accordingly.
            for i, f, w in izip(np.arange(nwalkers)[S1], factors, new_walkers):
                lnpdiff = f + w.lnprob - ensemble.walkers[i].lnprob
                if lnpdiff > np.log(ensemble.random.rand()):
                    ensemble.walkers[i] = w
                    ensemble.acceptance[i] = True

            # Update the ensemble with the accepted walkers.
            ensemble.update()

        # Do any move-specific cleanup.
        self.finalize(ensemble)

        return ensemble
