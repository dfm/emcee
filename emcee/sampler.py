# encoding: utf-8
"""
The base sampler class implementing various helpful functions.

"""

from __future__ import division

__all__ = ['Sampler']

import numpy as np

# import acor if it's available
try:
    import acor
except ImportError:
    acor = None

class Sampler(object):
    """
    An abstract sampler object that implements various helper functions

    #### Arguments

    * `dim` (int): Number of dimensions in the parameter space.
    * `lnpostfn` (callable): A function that takes a vector in the parameter
      space as input and returns the natural logarithm of the posterior
      probability for that position.

    #### Keyword Arguments

    * `args` (list): Optional list of extra arguments for `lnpostfn`.
      `lnpostfn` will be called with the sequence `lnpostfn(p, *args)`.

    """
    def __init__(self, dim, lnprobfn, args=[]):
        self.dim      = dim
        self.lnprobfn = lnprobfn
        self.args     = args

        # This is a random number generator that we can easily set the state
        # of without affecting the numpy-wide generator
        self._random = np.random.mtrand.RandomState()

        self.reset()

    @property
    def random_state(self):
        return self._random.get_state()

    @random_state.setter
    def random_state(self, state):
        try:
            self._random.set_state(state)
        except:
            pass

    @property
    def acceptance_fraction(self):
        return self.naccepted/self.iterations

    @property
    def chain(self):
        return self._chain

    @property
    def flatchain(self):
        return self._chain

    @property
    def lnprobability(self):
        return self._lnprob

    @property
    def acor(self):
        if acor is None:
            raise ImportError("acor")
        return acor.acor(self._chain.T)[0]

    def get_lnprob(self, p):
        return self.lnprobfn(p, *self.args)

    def reset(self):
        """Clear `chain`, `lnprobability` and the bookkeeping parameters."""
        self.iterations = 0
        self.naccepted  = 0

    def sample(self, *args, **kwargs):
        raise NotImplementedError("The sampling routine must be implemented "\
                "by subclasses")

    def run_mcmc(self, pos0, N, rstate0=None, lnprob0=None):
        """
        Iterate sample for `N` iterations and return the result. The arguments
        are passed directly to `sample` so see the parameter details given in
        `sample`.

        #### Returns

        The `(position, lnprob, state)` tuple after `N` iterations.

        """
        for pos,lnprob,state in self.sample(pos0, lnprob0, rstate0,
                                          iterations=N):
            pass
        return pos,lnprob,state


