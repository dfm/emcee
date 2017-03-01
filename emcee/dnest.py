from __future__ import print_function

__all__ = ["DNestSampler"]

import numpy as np

from .sampler import Sampler
from .ensemble import EnsembleSampler


class DNestLevels(object):

    def __init__(self, lnpriorfn, lnlikefn, lam=10.0):
        self.lnpriorfn = lnpriorfn
        self.lnlikefn = lnlikefn
        self.lstars = np.array([-np.inf])
        self.lam = lam

    def append(self, lstar):
        self.lstars = np.append(self.lstars, lstar)

    def __call__(self, p):
        pr = self.lnpriorfn(p)
        ll = self.lnlikefn(p)
        if np.isinf(pr):
            return -np.inf, ll
        J = len(self.lstars) - 1
        j = np.arange(J + 1)[self.lstars < ll][-1]
        w = np.exp((j - J) / self.lam)
        return w * pr, ll


class DNestSampler(Sampler):

    def __init__(self, nwalkers, dim, lnpriorfn, lnlikefn):
        super(DNestSampler, self).__init__(dim, lnpriorfn)
        self.levels = None
        self.nwalkers = nwalkers
        self.dim = dim
        self.lnpriorfn = lnpriorfn
        self.lnlikefn = lnlikefn

    def build_levels(self, p0, N, lam=1.0, nlevels=10, verbose=True):
        if verbose:
            print(u"Building levels...")
            print(u"{0:10s} {1:10s} {2:10s}".format(u"~log(X)",
                                                    u"log(L*)",
                                                    u"N_samps"))

        self.levels = DNestLevels(self.lnpriorfn, self.lnlikefn, lam=lam)
        sampler = EnsembleSampler(self.nwalkers, self.dim, self.levels)
        lstar = -np.inf
        ll = None
        for i in range(nlevels):
            sampler.reset()
            pos, lnprob, rstate, b = sampler.run_mcmc(p0, N)

            # Get the list of log likelihoods.
            blobs = np.array(sampler.blobs).T
            if ll is None:
                ll = blobs.flatten()
            else:
                ll = np.append(ll, blobs.flatten())

            # Trim the log-likelihood array to only include samples in the
            # current level.
            ll = ll[ll >= lstar]

            # Sort the likelihoods and get the top 1/e values.
            ll = np.sort(ll)[::-1]
            lstar = ll[np.floor(len(ll) * (1.0 / np.exp(1)))]
            self.levels.append(lstar)

            # Sample new positions making sure that they are in the next
            # level up.
            positions = sampler.chain
            xi = np.random.randint(0, self.nwalkers, size=self.nwalkers)
            yi = np.random.randint(0, N, size=self.nwalkers)
            p0 = positions[xi, yi, :]
            inds = blobs[xi, yi] < lstar
            while np.any(inds):
                n = np.sum(inds)
                xi[inds] = np.random.randint(0, self.nwalkers, size=n)
                yi[inds] = np.random.randint(0, N, size=n)
                p0[inds] = positions[xi[inds], yi[inds], :]
                inds = blobs[xi, yi] < lstar

            if verbose:
                print(u"{0:8d}   {1:6.3f} {2:10d}".format(-(i + 1),
                                                          lstar,
                                                          len(ll)))

    def explore(self, p0, N):
        js = np.zeros(self.nwalkers)
