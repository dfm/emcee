#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import numpy as np

from .integrator import leapfrog


def step_hmc(log_prob_fn, grad_log_prob_fn, metric, q, log_prob, epsilon,
             L):
    initial_q = np.array(q, copy=True)
    p = metric.sample_p()
    initial_h = 0.5 * np.dot(p, metric.dot(p))
    initial_h -= log_prob
    dUdq = -grad_log_prob_fn(q)
    for l in range(L):
        q, p, dUdq = leapfrog(grad_log_prob_fn, metric, q, p, epsilon,
                              dUdq)
    p = -p
    final_log_prob = log_prob_fn(q)
    final_h = 0.5 * np.dot(p, metric.dot(p))
    final_h -= final_log_prob
    accept = np.random.rand() < np.exp(initial_h - final_h)
    if accept:
        return q, final_log_prob, accept
    return initial_q, log_prob, accept


def simple_hmc(log_prob_fn, grad_log_prob_fn, q, niter, epsilon, L,
               metric=None):
    if metric is None:
        metric = IdentityMetric(len(q))

    samples = np.empty((niter, len(q)))
    samples_lp = np.empty(niter)
    log_prob = log_prob_fn(q)
    acc_count = 0
    for n in tqdm(range(niter), total=niter):
        q, log_prob, accept = step_hmc(log_prob_fn, grad_log_prob_fn,
                                       metric, q, log_prob, epsilon, L)
        acc_count += accept
        samples[n] = q
        samples_lp[n] = log_prob

    return samples, samples_lp, acc_count / float(niter)
