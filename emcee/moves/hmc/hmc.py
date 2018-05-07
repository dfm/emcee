# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["HamiltonianMove"]

import logging
from functools import partial

import numpy as np

from ..move import Move
from ...state import State
from .step_size import StepSize
from .integrator import leapfrog
from .metric import IdentityMetric


class HamiltonianMove(Move):
    """

    """

    def __init__(self, nstep, step_size=None, metric=None,
                 ntune=0, tune_step_size=True, tune_metric=True,
                 initial_buffer=100, final_buffer=100, window=25,
                 target_accept=0.8):
        self.nstep = int(nstep)
        if step_size is None:
            step_size = StepSize(delta=target_accept)
        elif not hasattr(step_size, "sample_step_size"):
            step_size = StepSize(step_size, delta=target_accept)
        self.step_size = step_size
        self.metric = metric

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
        """Tune the proposal parameters on a schedule

        Args:
            state (State): The current state of the ensemble.
            accepted (float): The average acceptance statistic of the current
                tree.

        """
        self.step_count += 1

        # Ignore if tuning is finished
        if self.step_count > self.ntune:
            return

        # At the end of tuning, the step size should be finalized
        elif self.step_count == self.ntune:
            if self.tune_step_size:
                self.step_size.finalize()
            return

        # Always update the step size
        if self.tune_step_size:
            self.step_size.update(np.mean(accepted))

        if self.tune_metric:
            # While inside a window, save the state to the metric
            if self.step_count >= self.initial_buffer and \
                    self.step_count < self.ntune - self.final_buffer:
                for vector in state.coords:
                    self.metric.update(vector)

            # At the end of a tuning window, finalize the metric and reset the
            # step size
            if self.step_count in self.windows:
                self.metric.finalize()
                if self.tune_step_size:
                    self.step_size.restart()

    def propose(self, model, state):
        if not callable(model.grad_log_prob_fn):
            raise ValueError("a grad_log_prob function must be provided to "
                             "use the Hamiltonian moves")
        if self.metric is None:
            self.metric = IdentityMetric(state.coords.shape[1])
        if state.coords.shape[1] != self.metric.ndim:
            raise ValueError("dimension mismatch between initial coordinates "
                             "and metric")
        nwalkers = state.coords.shape[0]

        # Sample the step sizes and generate new random number generators
        # first in order to deal with parallelization of the move
        steps = []
        randoms = []
        for i in range(nwalkers):
            steps.append(self.step_size.sample_step_size(random=model.random))
            randoms.append(
                np.random.RandomState(model.random.randint(2**16)))

        # Loop over walkers and run a step for each walker (possibly
        # in parallel)
        args = zip(state.coords, state.log_prob, steps, randoms)
        f = partial(self, model.log_prob_fn, model.grad_log_prob_fn)
        q, accepted = map(np.array, zip(*model.map_fn(f, args)))

        # Compute the probability at the final state
        new_log_probs, new_blobs = model.compute_log_prob_fn(q)
        new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
        state = self.update(state, new_state,
                            np.ones(state.coords.shape[0], dtype=bool))
        return state, accepted

    def __call__(self, log_prob_fn, grad_log_prob_fn, args):
        initial_q, initial_lp, epsilon, random = args
        q = initial_q
        p = self.metric.sample_p(random=random)

        H0 = 0.5 * np.dot(p, self.metric.dot(p))
        H0 -= initial_lp

        dUdq = -grad_log_prob_fn(q)
        for _ in range(self.nstep):
            q, p, dUdq = leapfrog(grad_log_prob_fn, self.metric,
                                  q, p, epsilon, dUdq)

        lp = log_prob_fn(q)
        try:
            lp = float(lp[0])
        except (IndexError, TypeError):
            pass

        H = 0.5 * np.dot(p, self.metric.dot(p))
        H -= lp
        accept = np.exp(H0 - H)
        if (not np.isfinite(H)) or accept < 1.0 and random.rand() > accept:
            q = initial_q

        return q, accept
