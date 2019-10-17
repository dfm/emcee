# -*- coding: utf-8 -*-

from collections import namedtuple

__all__ = ["Model"]


Model = namedtuple(
    "Model", ("log_prob_fn", "compute_log_prob_fn", "map_fn", "random")
)
