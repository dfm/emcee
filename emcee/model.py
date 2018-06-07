# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Model"]

from collections import namedtuple

Model = namedtuple(
    "Model",
    ("log_prob_fn", "compute_log_prob_fn", "map_fn", "random")
)
