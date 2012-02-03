# encoding: utf-8
"""
A vanilla Metropolis-Hastings sampler

"""

__all__ = ['MHSampler']

import numpy as np

from sampler import Sampler

# === MHSampler ===
class MHSampler(Sampler):
    """
    The most basic possible Metropolis-Hastings style MCMC sampler

    #### Arguments

    * `cov` (numpy.ndarray): The covariance matrix to use for the proposal
      distribution.
    * `dim` (int): Number of dimensions in the parameter space.
    * `lnpostfn` (callable): A function that takes a vector in the parameter
      space as input and returns the natural logarithm of the posterior
      probability for that position.

    #### Keyword Arguments

    * `args` (list): Optional list of extra arguments for `lnpostfn`.
      `lnpostfn` will be called with the sequence `lnpostfn(p, *args)`.

    #### Notes

    The 'chain' member of this object has the shape: (nlinks, dim) where 'nlinks'
    is the number of steps taken by the chain.

    """
    def __init__(self, cov, *args, **kwargs):
        super(MHSampler, self).__init__(*args, **kwargs)
        self.cov = cov

    def reset(self):
        super(MHSampler, self).reset()
        self._chain  = np.empty((0, self.dim))
        self._lnprob = np.empty(0)

    def sample(self, p0, lnprob=None, randomstate=None, storechain=True, resample=1,
            iterations=1):
        """
        Advances the chain iterations steps as an iterator

        #### Arguments

        * `pos0` (numpy.ndarray): The initial position vector.

        #### Keyword Arguments

        * `lnprob0` (float): The log posterior probability at position `p0`.
          If `lnprob is None`, the initial value is calculated.
        * `rstate0` (tuple): The state of the random number generator.
          See the `Sampler.random_state` property for details.
        * `iterations` (int): The number of steps to run. (default: 1)

        #### Yields

        * `pos` (numpy.ndarray): The final position vector.
        * `lnprob` (float): The log-probability at `pos`.
        * `rstate` (tuple): The state of the random number generator.

        """
        self.random_state = randomstate

        p = np.array(p0)
        if lnprob is None:
            lnprob = self.get_lnprob(p)

        # Resize the chain in advance.
        if storechain:
            N = int(iterations/resample)
            self._chain = np.concatenate((self._chain,
                    np.zeros((N, self.dim))), axis=0)
            self._lnprob = np.append(self._lnprob, np.zeros(N))

        i0 = self.iterations
        for i in xrange(int(iterations)):
            self.iterations += 1

            # Calculate the proposal distribution.
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

            # Heavy duty iterator action going on right here...
            yield p, lnprob, self.random_state

