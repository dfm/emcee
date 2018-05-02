# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["NUTSMove"]

import logging
from collections import namedtuple

import numpy as np

from ..move import Move
from ...state import State
from .step_size import StepSize
from .integrator import leapfrog
from .metric import IdentityMetric


Point = namedtuple("Point", ("q", "p", "U", "dUdq"))


def _nuts_criterion(p_sharp_minus, p_sharp_plus, rho):
    return np.dot(p_sharp_plus, rho) > 0 and np.dot(p_sharp_minus, rho) > 0


def _nuts_tree(log_prob_fn, grad_log_prob_fn, metric, epsilon,
               depth, z, z_propose, p_sharp_left, p_sharp_right, rho, H0,
               sign, nleapfrog, log_sum_weight, sum_metro_prob, max_depth,
               max_delta_h, random):
    if depth == 0:
        q, p, dUdq = leapfrog(grad_log_prob_fn, metric, z.q, z.p,
                              sign * epsilon, z.dUdq)
        z = Point(q, p, -log_prob_fn(q), dUdq)
        nleapfrog += 1

        h = 0.5 * np.dot(p, metric.dot(p))
        h += z.U
        if not np.isfinite(h):
            h = np.inf
        valid_subtree = (h - H0) <= max_delta_h

        log_sum_weight = np.logaddexp(log_sum_weight, H0 - h)
        sum_metro_prob += min(np.exp(H0 - h), 1.0)

        z_propose = z
        rho += z.p

        p_sharp_left = metric.dot(z.p)
        p_sharp_right = p_sharp_left

        return (
            valid_subtree, z, z_propose, p_sharp_left, p_sharp_right, rho,
            nleapfrog, log_sum_weight, sum_metro_prob
        )

    p_sharp_dummy = np.empty_like(p_sharp_left)

    # Left
    log_sum_weight_left = -np.inf
    rho_left = np.zeros_like(rho)
    results_left = _nuts_tree(
        log_prob_fn, grad_log_prob_fn, metric, epsilon,
        depth - 1, z, z_propose, p_sharp_left, p_sharp_dummy, rho_left,
        H0, sign, nleapfrog, log_sum_weight_left, sum_metro_prob, max_depth,
        max_delta_h, random
    )
    (valid_left, z, z_propose, p_sharp_left, p_sharp_dummy, rho_left,
     nleapfrog, log_sum_weight_left, sum_metro_prob) = results_left

    if not valid_left:
        return (
            False, z, z_propose, p_sharp_left, p_sharp_right, rho,
            nleapfrog, log_sum_weight, sum_metro_prob
        )

    # Right
    z_propose_right = Point(z.q, z.p, z.U, z.dUdq)
    log_sum_weight_right = -np.inf
    rho_right = np.zeros_like(rho)
    results_right = _nuts_tree(
        log_prob_fn, grad_log_prob_fn, metric, epsilon,
        depth - 1, z, z_propose_right, p_sharp_dummy, p_sharp_right, rho_right,
        H0, sign, nleapfrog, log_sum_weight_right, sum_metro_prob, max_depth,
        max_delta_h, random
    )
    (valid_right, z, z_propose_right, p_sharp_dummy, p_sharp_right, rho_right,
     nleapfrog, log_sum_weight_right, sum_metro_prob) = results_right

    if not valid_right:
        return (
            False, z, z_propose, p_sharp_left, p_sharp_right, rho,
            nleapfrog, log_sum_weight, sum_metro_prob
        )

    # Multinomial sample from the right
    log_sum_weight_subtree = np.logaddexp(log_sum_weight_left,
                                          log_sum_weight_right)
    log_sum_weight = np.logaddexp(log_sum_weight, log_sum_weight_subtree)

    if log_sum_weight_right > log_sum_weight_subtree:
        z_propose = z_propose_right
    else:
        accept_prob = np.exp(log_sum_weight_right - log_sum_weight_subtree)
        if random.rand() < accept_prob:
            z_propose = z_propose_right

    rho_subtree = rho_left + rho_right
    rho += rho_subtree

    return (
        _nuts_criterion(p_sharp_left, p_sharp_right, rho_subtree),
        z, z_propose, p_sharp_left, p_sharp_right, rho,
        nleapfrog, log_sum_weight, sum_metro_prob
    )


def step_nuts(log_prob_fn, grad_log_prob_fn, metric, q, log_prob,
              epsilon, max_depth, max_delta_h, random):
    dUdq = -grad_log_prob_fn(q)
    p = metric.sample_p(random=random)

    z_plus = Point(q, p, -log_prob, dUdq)
    z_minus = Point(q, p, -log_prob, dUdq)
    z_sample = Point(q, p, -log_prob, dUdq)
    z_propose = Point(q, p, -log_prob, dUdq)

    p_sharp_plus = metric.dot(p)
    p_sharp_dummy = np.array(p_sharp_plus, copy=True)
    p_sharp_minus = np.array(p_sharp_plus, copy=True)
    rho = np.array(p, copy=True)

    nleapfrog = 0
    log_sum_weight = 0.0
    sum_metro_prob = 0.0
    H0 = 0.5 * np.dot(p, metric.dot(p))
    H0 -= log_prob

    for depth in range(max_depth):
        rho_subtree = np.zeros_like(rho)
        valid_subtree = False
        log_sum_weight_subtree = -np.inf

        if random.rand() > 0.5:
            results = _nuts_tree(
                log_prob_fn, grad_log_prob_fn, metric, epsilon,
                depth, z_plus, z_propose, p_sharp_dummy, p_sharp_plus,
                rho_subtree, H0, 1, nleapfrog, log_sum_weight_subtree,
                sum_metro_prob, max_depth, max_delta_h, random)
            (valid_subtree, z_plus, z_propose, p_sharp_dummy, p_sharp_plus,
             rho_subtree, nleapfrog, log_sum_weight_subtree, sum_metro_prob) \
                = results

        else:
            results = _nuts_tree(
                log_prob_fn, grad_log_prob_fn, metric, epsilon,
                depth, z_minus, z_propose, p_sharp_dummy, p_sharp_minus,
                rho_subtree, H0, -1, nleapfrog, log_sum_weight_subtree,
                sum_metro_prob, max_depth, max_delta_h, random)
            (valid_subtree, z_minus, z_propose, p_sharp_dummy, p_sharp_minus,
             rho_subtree, nleapfrog, log_sum_weight_subtree, sum_metro_prob) \
                = results

        if not valid_subtree:
            break

        if log_sum_weight_subtree > log_sum_weight:
            z_sample = z_propose
        else:
            accept_prob = np.exp(log_sum_weight_subtree - log_sum_weight)
            if random.rand() < accept_prob:
                z_sample = z_propose

        log_sum_weight = np.logaddexp(log_sum_weight, log_sum_weight_subtree)
        rho += rho_subtree

        if not _nuts_criterion(p_sharp_minus, p_sharp_plus, rho):
            break

    accept_prob = sum_metro_prob / nleapfrog
    return z_sample.q, float(accept_prob)


class NUTSMove(Move):

    def __init__(self, step_size=None, metric=None, max_depth=5,
                 max_delta_h=1000.0,
                 ntune=0, tune_step_size=True, tune_metric=True,
                 initial_buffer=100, final_buffer=100, window=25,
                 parallel_safe=True):
        if step_size is None:
            step_size = StepSize()
        elif not hasattr(step_size, "sample_step_size"):
            step_size = StepSize(step_size)
        self.step_size = step_size
        self.metric = metric
        self.max_depth = max_depth
        self.max_delta_h = max_delta_h
        self.parallel_safe = parallel_safe

        self.ntune = max(int(ntune), 0)
        self.tune_metric = tune_metric
        self.tune_step_size = tune_step_size
        if ntune > 0 and tune_metric:
            # First, how many inner steps do we get?
            self.initial_buffer = max(int(initial_buffer), 0)
            self.final_buffer = max(int(final_buffer), 0)
            ninner = self.ntune - self.initial_buffer - self.final_buffer
            if ninner <= window:
                logging.warn("not enough tuning samples for proposed "
                             "schedule; resizing to 20%/70%/10%")
                self.initial_buffer, ninner, self.final_buffer = \
                    (np.array([0.2, 0.7, 0.1]) * self.ntune).astype(int)

            # Compute the tuning schedule
            p = max(1, np.ceil(np.log2(ninner) - np.log2(window)) + 1)
            windows = window * 2 ** np.arange(p)
            if len(windows) <= 1:
                windows = np.array([ninner])
            else:
                if windows[-1] > ninner:
                    windows = np.append(windows[:-2], ninner)

            self.windows = set((np.append(0, windows) +
                                self.initial_buffer).astype(int))
        else:
            self.initial_buffer = 0
            self.windows = set()

        self.step_count = 0

    def tune(self, state, accepted):
        self.step_count += 1
        if self.step_count > self.ntune:
            return
        elif self.step_count == self.ntune:
            if self.tune_step_size:
                self.step_size.finalize()
            return

        if self.tune_step_size:
            self.step_size.update(np.mean(accepted))

        if self.tune_metric:
            if self.step_count >= self.initial_buffer and \
                    self.step_count < self.ntune - self.final_buffer:
                for vector in state.coords:
                    self.metric.update(vector)

            if self.step_count in self.windows:
                self.metric.finalize()
                if self.tune_step_size:
                    self.step_size.restart()

    def propose(self, model, state):
        if not callable(model.grad_log_prob_fn):
            raise ValueError("a grad_log_prob function must be provided to "
                             "use the NUTSMove")

        if self.metric is None:
            self.metric = IdentityMetric(state.coords.shape[1])
        if state.coords.shape[1] != self.metric.ndim:
            raise ValueError("dimension mismatch between initial coordinates "
                             "and metric")

        nwalkers = state.coords.shape[0]
        q = np.empty_like(state.coords)
        accepted = np.zeros(nwalkers)
        for i in range(nwalkers):
            # Sample the step size including jitter
            step = self.step_size.sample_step_size(random=model.random)

            if self.parallel_safe:
                random = np.random.RandomState(model.random.randint(2**32))
            else:
                random = model.random

            # Run one step of NUTS
            q[i], accepted[i] = step_nuts(
                model.log_prob_fn, model.grad_log_prob_fn, self.metric,
                state.coords[i], state.log_prob[i], step,
                self.max_depth, self.max_delta_h, random)

        new_log_probs, new_blobs = model.compute_log_prob_fn(q)
        new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
        state = self.update(state, new_state,
                            np.ones(state.coords.shape[0], dtype=bool))
        return state, accepted
