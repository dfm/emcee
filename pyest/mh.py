# encoding: utf-8
"""
A vanilla Metropolis-Hastings sampler

"""

from __future__ import division

__all__ = ['MHSampler']

import numpy as np

from sampler import Sampler

class MHSampler(Sampler):
    """
    The most basic possible Metropolis-Hastings style MCMC sampler for comparison

    Parameters
    ----------
    cov : numpy.ndarray (dim, dim)
        The covariance matrix to use for the proposal distribution.

    dim : int
        The dimension of the parameter space.

    lnprobfn : callable
        A function that computes the probability of a particular point in phase
        space.  Will be called as lnprobfn(p, *args)

    args : list, optional
        A list of arguments for lnprobfn.

    Notes
    -----
    The 'chain' member of this object has the shape: (nlinks, dim) where 'nlinks'
    is the number of steps taken by the chain.

    """
    def __init__(self, cov, *args, **kwargs):
        super(MHSampler, self).__init__(*args, **kwargs)
        self.cov = cov

    def do_reset(self):
        self._chain  = np.empty((0, self.dim))
        self._lnprob = np.empty(0)

    def sample(self, p0, lnprob=None, randomstate=None, storechain=True, resample=1,
            iterations=1):
        self.random_state = randomstate

        p = np.array(p0)
        if lnprob is None:
            lnprob = self.get_lnprob(p)

        # resize chain
        if storechain:
            N = int(iterations/resample)
            self._chain = np.concatenate((self._chain,
                    np.zeros((N, self.dim))), axis=0)
            self._lnprob = np.append(self._lnprob, np.zeros(N))

        i0 = self.iterations
        for i in xrange(int(iterations)):
            self.iterations += 1

            # proposal
            q = self._random.multivariate_normal(p, self.cov)
            newlnprob = self.get_lnprob(q)
            diff = newlnprob-lnprob

            # M-H acceptance ratio
            if diff < 0:
                diff = np.exp(diff) - self._random.rand()

            if diff > 0:
                p = q
                lnprob = newlnprob
                self.naccepted += 1

            if storechain and i%resample == 0:
                ind = i0 + int(i/resample)
                self._chain[ind,:] = p
                self._lnprob[ind]  = lnprob

            # heavy duty iterator action going on right here
            yield p, lnprob, self.random_state

