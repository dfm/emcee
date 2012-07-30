# encoding: utf-8
"""
The base sampler class implementing various helpful functions.

"""

from __future__ import division

__all__ = ["Sampler"]

import numpy as np

try:
    import acor
    acor = acor
except ImportError:
    acor = None


class Sampler(object):
    """
    An abstract sampler object that implements various helper functions

    :param dim:
        The number of dimensions in the parameter space.

    :param lnpostfn:
        A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the posterior probability for that
        position.

    :param args: (optional)
        A list of extra arguments for ``lnpostfn``. ``lnpostfn`` will be
        called with the sequence ``lnpostfn(p, *args)``.

    """
    def __init__(self, dim, lnprobfn, args=[]):
        self.dim = dim
        self.lnprobfn = lnprobfn
        self.args = args

        # This is a random number generator that we can easily set the state
        # of without affecting the numpy-wide generator
        self._random = np.random.mtrand.RandomState()

        self.reset()

    @property
    def random_state(self):
        """
        The state of the internal random number generator. In practice, it's
        the result of calling ``get_state()`` on a
        ``numpy.random.mtrand.RandomState`` object. You can try to set this
        property but be warned that if you do this and it fails, it will do
        so silently.

        """
        return self._random.get_state()

    @random_state.setter
    def random_state(self, state):
        """
        Try to set the state of the random number generator but fail silently
        if it doesn't work. Don't say I didn't warn you...

        """
        try:
            self._random.set_state(state)
        except:
            pass

    @property
    def acceptance_fraction(self):
        """
        The fraction of proposed steps that were accepted.

        """
        return self.naccepted / self.iterations

    @property
    def chain(self):
        """
        A pointer to the Markov chain.

        """
        return self._chain

    @property
    def flatchain(self):
        """
        Alias of ``chain`` provided for compatibility.

        """
        return self._chain

    @property
    def lnprobability(self):
        """
        A list of the log-probability values associated with each step in
        the chain.

        """
        return self._lnprob

    @property
    def acor(self):
        """
        The autocorrelation time of each parameter in the chain (length:
        ``dim``) as estimated by the ``acor`` module.

        """
        if acor is None:
            raise ImportError("acor")
        return acor.acor(self._chain.T)[0]

    def get_lnprob(self, p):
        """Return the log-probability at the given position."""
        return self.lnprobfn(p, *self.args)

    def reset(self):
        """
        Clear ``chain``, ``lnprobability`` and the bookkeeping parameters.

        """
        self.iterations = 0
        self.naccepted = 0

    def clear_chain(self):
        """An alias for :func:`reset` kept for backwards compatibility."""
        return self.reset()

    def sample(self, *args, **kwargs):
        raise NotImplementedError("The sampling routine must be implemented "\
                "by subclasses")

    def run_mcmc(self, pos0, N, rstate0=None, lnprob0=None, **kwargs):
        """
        Iterate :func:`sample` for ``N`` iterations and return the result.

        :param p0:
            The initial position vector.

        :param N:
            The number of steps to run.

        :param lnprob0: (optional)
            The log posterior probability at position ``p0``. If ``lnprob``
            is not provided, the initial value is calculated.

        :param rstate0: (optional)
            The state of the random number generator. See the
            :func:`random_state` property for details.

        :param **kwargs: (optional)
            Other parameters that are directly passed to :func:`sample`.

        """
        for results in self.sample(pos0, lnprob0, rstate0, iterations=N,
                **kwargs):
            pass
        return results
