# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["State"]

import numpy as np


class State(object):

    __slots__ = "coords", "log_prob", "blobs"

    def __init__(self, coords, log_prob=None, blobs=None):
        if hasattr(coords, "coords"):
            self.coords = coords.coords
            self.log_prob = coords.log_prob
            self.blobs = coords.blobs
            return

        self.coords = np.atleast_2d(coords)
        self.log_prob = log_prob
        self.blobs = blobs
