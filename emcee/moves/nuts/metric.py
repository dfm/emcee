# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["IdentityMetric", "IsotropicMetric", "DiagonalMetric",
           "DenseMetric"]

import numpy as np
from scipy.linalg import cholesky, solve_triangular


class IdentityMetric(object):

    def __init__(self, ndim):
        self.ndim = int(ndim)

    def update_variance(self, variance):
        pass

    def sample_p(self, random=None):
        if random is None:
            random = np.random
        return random.randn(self.ndim)

    def dot(self, p):
        return p

    def restart(self):
        pass

    def update(self, sample):
        pass

    def finalize(self):
        pass


class IsotropicMetric(IdentityMetric):

    def __init__(self, ndim, variance=1.0):
        self.ndim = int(ndim)
        self.variance = float(variance)

    def update_variance(self, variance):
        self.variance = variance

    def sample_p(self, random=None):
        if random is None:
            random = np.random
        return random.randn(self.ndim) / np.sqrt(self.variance)

    def dot(self, p):
        return p * self.variance


class DiagonalMetric(IsotropicMetric):

    def __init__(self, variance):
        self.ndim = len(variance)
        self.variance = variance
        self.restart()

    def restart(self):
        self.counter = 0
        self.m = np.zeros(self.ndim)
        self.m2 = np.zeros(self.ndim)

    def update(self, sample):
        self.counter += 1
        delta = sample - self.m
        self.m += delta / self.counter
        self.m2 += (sample - self.m) * delta

    def finalize(self):
        if self.counter < 1:
            return
        var = self.m2 / (self.counter - 1)
        n = self.counter
        self.variance = (n / (n + 5.0)) * var
        self.variance += 1e-3 * (5.0 / (n + 5.0))
        self.restart()


class DenseMetric(IdentityMetric):

    def __init__(self, variance):
        self.ndim = len(variance)
        self.update_variance(variance)
        self.restart()

    def update_variance(self, variance):
        self.L = cholesky(variance, lower=False)
        self.variance = variance

    def sample_p(self, random=None):
        if random is None:
            random = np.random
        return solve_triangular(self.L, random.randn(self.ndim),
                                lower=False)

    def dot(self, p):
        return np.dot(self.variance, p)

    def restart(self):
        self.counter = 0
        self.m = np.zeros(self.ndim)
        self.m2 = np.zeros((self.ndim, self.ndim))

    def update(self, sample):
        self.counter += 1
        delta = sample - self.m
        self.m += delta / self.counter
        self.m2 += (sample - self.m)[:, None] * delta[None, :]

    def finalize(self):
        if self.counter < 1:
            return
        cov = self.m2 / (self.counter - 1)
        n = self.counter
        cov *= (n / (n + 5.0))
        cov[np.diag_indices_from(cov)] += 1e-3 * (5.0 / (n + 5.0))
        self.update_variance(cov)
        self.restart()
