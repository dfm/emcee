# -*- coding: utf-8 -*-

# The code in this module is a port of the nuts_base proposal from the Stan
# project to Python. Stan is licensed under the following license:
#
# BSD 3-Clause License
#
# Copyright (c) 2011--2017, Stan Developers and their Assignees All rights
# reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import division, print_function

__all__ = ["NoUTurnMove"]

from collections import namedtuple

import numpy as np

from .integrator import leapfrog
from .hmc import HamiltonianMove


Point = namedtuple("Point", ("q", "p", "U", "dUdq"))


class NoUTurnMove(HamiltonianMove):
    """

    """

    def __init__(self, max_depth=5, max_delta_h=1000.0, **kwargs):
        self.max_depth = max_depth
        self.max_delta_h = max_delta_h
        super(NoUTurnMove, self).__init__(0, **kwargs)

    def __call__(self, log_prob_fn, grad_log_prob_fn, args):
        q, log_prob, epsilon, random = args

        dUdq = -grad_log_prob_fn(q)
        p = self.metric.sample_p(random=random)

        z_plus = Point(q, p, -log_prob, dUdq)
        z_minus = Point(q, p, -log_prob, dUdq)
        z_sample = Point(q, p, -log_prob, dUdq)
        z_propose = Point(q, p, -log_prob, dUdq)

        p_sharp_plus = self.metric.dot(p)
        p_sharp_dummy = np.array(p_sharp_plus, copy=True)
        p_sharp_minus = np.array(p_sharp_plus, copy=True)
        rho = np.array(p, copy=True)

        nleapfrog = 0
        log_sum_weight = 0.0
        sum_metro_prob = 0.0
        H0 = 0.5 * np.dot(p, self.metric.dot(p))
        H0 -= log_prob

        for depth in range(self.max_depth):
            rho_subtree = np.zeros_like(rho)
            valid_subtree = False
            log_sum_weight_subtree = -np.inf

            if random.rand() > 0.5:
                results = self.build_tree(
                    log_prob_fn, grad_log_prob_fn, nleapfrog, depth, 1,
                    epsilon,
                    z_plus, z_propose, p_sharp_dummy, p_sharp_plus,
                    rho_subtree, H0,
                    log_sum_weight_subtree, sum_metro_prob, random)
                (valid_subtree, z_plus, z_propose, p_sharp_dummy, p_sharp_plus,
                 rho_subtree, nleapfrog, log_sum_weight_subtree,
                 sum_metro_prob) = results

            else:
                results = self.build_tree(
                    log_prob_fn, grad_log_prob_fn, nleapfrog, depth, -1,
                    epsilon,
                    z_minus, z_propose, p_sharp_dummy, p_sharp_minus,
                    rho_subtree, H0,
                    log_sum_weight_subtree, sum_metro_prob, random)
                (valid_subtree, z_minus, z_propose, p_sharp_dummy,
                 p_sharp_minus, rho_subtree, nleapfrog, log_sum_weight_subtree,
                 sum_metro_prob) = results

            if not valid_subtree:
                break

            if log_sum_weight_subtree > log_sum_weight:
                z_sample = z_propose
            else:
                accept_prob = np.exp(log_sum_weight_subtree - log_sum_weight)
                if random.rand() < accept_prob:
                    z_sample = z_propose

            log_sum_weight = np.logaddexp(log_sum_weight,
                                          log_sum_weight_subtree)
            rho += rho_subtree

            if not self.compute_criterion(p_sharp_minus, p_sharp_plus, rho):
                break

        accept_prob = sum_metro_prob / nleapfrog
        return z_sample.q, float(accept_prob)

    def build_tree(self, log_prob_fn, grad_log_prob_fn, nleapfrog, depth,
                   sign, epsilon,
                   z, z_propose, p_sharp_left, p_sharp_right, rho, H0,
                   log_sum_weight, sum_metro_prob, random):
        if depth == 0:
            q, p, dUdq = leapfrog(grad_log_prob_fn, self.metric,
                                  z.q, z.p, sign * epsilon, z.dUdq)
            lp = log_prob_fn(q)
            try:
                lp = float(lp[0])
            except (IndexError, TypeError):
                pass
            z = Point(q, p, -lp, dUdq)
            nleapfrog += 1

            h = 0.5 * np.dot(p, self.metric.dot(p))
            h += z.U
            if not np.isfinite(h):
                h = np.inf
            valid_subtree = (h - H0) <= self.max_delta_h

            log_sum_weight = np.logaddexp(log_sum_weight, H0 - h)
            sum_metro_prob += min(np.exp(H0 - h), 1.0)

            z_propose = z
            rho += z.p

            p_sharp_left = self.metric.dot(z.p)
            p_sharp_right = p_sharp_left

            return (
                valid_subtree, z, z_propose, p_sharp_left, p_sharp_right, rho,
                nleapfrog, log_sum_weight, sum_metro_prob
            )

        p_sharp_dummy = np.empty_like(p_sharp_left)

        # Left
        log_sum_weight_left = -np.inf
        rho_left = np.zeros_like(rho)
        results_left = self.build_tree(
            log_prob_fn, grad_log_prob_fn, nleapfrog, depth-1, sign, epsilon,
            z, z_propose, p_sharp_left, p_sharp_dummy, rho_left, H0,
            log_sum_weight_left, sum_metro_prob, random)
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
        results_right = self.build_tree(
            log_prob_fn, grad_log_prob_fn, nleapfrog, depth-1, sign, epsilon,
            z, z_propose_right, p_sharp_dummy, p_sharp_right, rho_right, H0,
            log_sum_weight_right, sum_metro_prob, random)
        (valid_right, z, z_propose_right, p_sharp_dummy, p_sharp_right,
         rho_right, nleapfrog, log_sum_weight_right, sum_metro_prob) = \
            results_right

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
            self.compute_criterion(p_sharp_left, p_sharp_right, rho_subtree),
            z, z_propose, p_sharp_left, p_sharp_right, rho,
            nleapfrog, log_sum_weight, sum_metro_prob
        )

    def compute_criterion(self, p_sharp_minus, p_sharp_plus, rho):
        return np.dot(p_sharp_plus, rho) > 0 and np.dot(p_sharp_minus, rho) > 0
