# encoding: utf-8
"""
The base sampler class implementing various helpful functions.

"""

from __future__ import division

__all__ = ['Sampler']

import numpy as np

import acor

class Sampler(object):
    """
    The base sampler object that implements various helper functions

    **Arguments**

    * `dim` (int): Number of dimensions in the parameter space.
    * `lnpostfn` (callable): A function that takes a vector in the parameter
      space as input and returns the natural logarithm of the posterior
      probability for that position.

    **Keyword Arguments**

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

    def reset(self):
        """Clear `chain`, `lnprobability` and the bookkeeping parameters."""
        self.iterations = 0
        self.naccepted  = 0

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
        return acor.acor(self._chain.T)[0]

    def get_lnprob(self, p):
        return self.lnprobfn(p, *self.args)

