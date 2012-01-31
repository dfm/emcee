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

    Parameters
    ----------
    dim : int
        The dimension of the parameter space.

    lnprobfn : callable
        A function that computes the probability of a particular point in phase
        space.  Will be called as lnprobfn(p, *args)

    args : list, optional
        A list of arguments for lnprobfn.

    """
    def __init__(self, dim, lnprobfn, args=[]):
        self.dim      = dim
        self.lnprobfn = lnprobfn
        self.args     = args

        # Initialize a random number generator that we own
        self._random = np.random.mtrand.RandomState()

        self.reset()

    def reset(self):
        """
        Reset the chain parameters (e.g. after the burn-in).

        """
        self.iterations = 0
        self.naccepted  = 0

        self.do_reset()

    def do_reset(self):
        """Implemented by subclasses"""
        pass

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

