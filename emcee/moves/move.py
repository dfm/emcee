# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

__all__ = ["Move"]


class Move(object):

    def update(self,
               coords, log_probs, blobs,
               new_coords, new_log_probs, new_blobs,
               accepted, subset=None):
        if subset is None:
            subset = np.ones(len(coords), dtype=bool)
        inds = np.arange(len(coords))
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
            for i, j in zip(inds[m1], np.arange(len(new_blobs))[m2]):
                blobs[i] = new_blobs[j]

        return coords, log_probs, blobs
