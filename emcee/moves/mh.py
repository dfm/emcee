# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .move import Move

__all__ = ["MHMove"]


class MHMove(Move):
    """
    A general Metropolis-Hastings proposal.

    :param proposal:
        The proposal function. It should take 2 arguments: a numpy-compatible
        random number generator and a ``(K, ndim)`` list of coordinate
        vectors. This function should return the proposed position and the
        log-ratio of the proposal probabilities (:math:`\ln q(x;\,x^\prime) -
        \ln q(x^\prime;\,x)` where :math:`x^\prime` is the proposed
        coordinate).

    :param ndim: (optional)
        If this proposal is only valid for a specific dimension of parameter
        space, set that here.

    """
    def __init__(self, proposal_function, ndim=None):
        self.ndim = ndim
        self.get_proposal = proposal_function

    def propose(self, coords, log_probs, blobs, log_prob_fn, random):
        """
        :param coords:
            The initial coordinates of the walkers.

        :param log_probs:
            The initial log probabilities of the walkers.

        :param log_prob_fn:
            A function that computes the log probabilities for a subset of
            walkers.

        :param random:
            A numpy-compatible random number state.

        """
        # Check to make sure that the dimensions match.
        nwalkers, ndim = coords.shape
        if self.ndim is not None and self.ndim != ndim:
            raise ValueError("Dimension mismatch in proposal")

        # Get the move-specific proposal.
        q, factors = self.get_proposal(coords, random)

        # Compute the lnprobs of the proposed position.
        new_log_probs, new_blobs = log_prob_fn(q)

        # Loop over the walkers and update them accordingly.
        lnpdiff = new_log_probs - log_probs + factors
        accepted = np.log(random.rand(nwalkers)) < lnpdiff

        # Update the parameters
        coords, log_probs, blobs = self.update(
            coords, log_probs, blobs,
            q, new_log_probs, new_blobs,
            accepted)

        return coords, log_probs, blobs, accepted
