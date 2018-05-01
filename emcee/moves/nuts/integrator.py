# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["leapfrog"]

import numpy as np


def leapfrog(grad_log_prob_fn, metric, q, p, epsilon, dUdq=None):
    q = np.array(q, copy=True)
    p = np.array(p, copy=True)

    if dUdq is None:
        dUdq = -grad_log_prob_fn(q)
    p -= 0.5 * epsilon * dUdq
    dTdp = metric.dot(p)
    q += epsilon * dTdp
    dUdq = -grad_log_prob_fn(q)
    p -= 0.5 * epsilon * dUdq

    return q, p, dUdq
