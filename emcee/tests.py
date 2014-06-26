# encoding: utf-8

import numpy as np

from . import proposals
from .sampler import Sampler


logprecision = -4


def lnprob_gaussian(x, icov):
    return -np.dot(x, np.dot(icov, x)) / 2.0


class Tests:

    def setUp(self):
        self.nwalkers = 100
        self.ndim = 5

        self.ntemp = 20

        self.N = 1000

        self.mean = np.zeros(self.ndim)
        self.cov = 0.5 - np.random.rand(self.ndim ** 2) \
            .reshape((self.ndim, self.ndim))
        self.cov = np.triu(self.cov)
        self.cov += self.cov.T - np.diag(self.cov.diagonal())
        self.cov = np.dot(self.cov, self.cov)
        self.icov = np.linalg.inv(self.cov)
        self.p0 = [0.1 * np.random.randn(self.ndim)
                   for i in range(self.nwalkers)]
        self.truth = np.random.multivariate_normal(self.mean, self.cov, 100000)

    def test_gaussian_proposal(self):
        self.sampler =
