# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

__all__ = ["Move"]


class Move(object):

    def update(self,
               coords, log_probs, blobs,
               new_coords, new_log_probs, new_blobs,
               accepted, subset=None):
        """Update a given subset of the ensemble with an accepted proposal

        Args:
            coords: The original ensemble coordinates.
            log_probs: The original log probabilities of the walkers.
            blobs: The original blobs.
            new_coords: The proposed coordinates.
            new_log_probs: The proposed log probabilities.
            new_blobs: The proposed blobs.
            accepted: A vector of booleans indicating which walkers were
                accepted.
            subset (Optional): A boolean mask indicating which walkers were
                included in the subset. This can be used, for example, when
                updating only the primary ensemble in a :class:`RedBlueMove`.

        """
        if subset is None:
            subset = np.ones(len(coords), dtype=bool)
        m1 = subset & accepted
        m2 = accepted[subset]
        coords[m1] = new_coords[m2]
        log_probs[m1] = new_log_probs[m2]

        if new_blobs is not None:
            if blobs is None:
                raise ValueError(
                    "If you start sampling with a given lnprob, "
                    "you also need to provide the current list of "
                    "blobs at that position.")
            blobs[m1] = new_blobs[m2]

        return coords, log_probs, blobs
