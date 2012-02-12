# encoding: utf-8
"""
An affine invariant Markov chain Monte Carlo (MCMC) sampler.

Goodman & Weare, Ensemble Samplers With Affine Invariance
   Comm. App. Math. Comp. Sci., Vol. 5 (2010), No. 1, 65–80

"""

__all__ = ['EnsembleSampler']

import multiprocessing
import numpy as np

try:
    import acor
except ImportError:
    acor = None

from sampler import Sampler

# === EnsembleSampler ===
class EnsembleSampler(Sampler):
    """
    A generalized Ensemble sampler that uses 2 ensembles for parallelization.

    This is a subclass of the `Sampler` object. See the `Sampler` object
    for more details about the inherited properties.

    #### Arguments

    * `k` (int): The number of Goodman & Weare "walkers".
    * `dim` (int): Number of dimensions in the parameter space.
    * `lnpostfn` (callable): A function that takes a vector in the parameter
      space as input and returns the natural logarithm of the posterior
      probability for that position.

    #### Keyword Arguments

    * `a` (float): The proposal scale parameter. (default: `2.0`)
    * `args` (list): Optional list of extra arguments for `lnpostfn`.
      `lnpostfn` will be called with the sequence `lnpostfn(p, *args)`.
    * `postargs` (list): Alias of `args` for backwards compatibility.
    * `threads` (int): The number of threads to use for parallelization.
      If `threads == 1`, then the `multiprocessing` module is not used but if
      `threads > 1`, then a `Pool` object is created and calls to `lnpostfn`
      are run in parallel. (default: 1)
    * `pool` (multiprocessing.Pool): An alternative method of using the
      parallelized algorithm. If `pool is not None`, the value of `threads`
      is ignored and the provided `Pool` is used for all parallelization.
      (default: `None`)

    #### Exceptions

    * `AssertionError`: If `k < 2*dim` or if `k` is not even.

    #### Warning

    The `chain` member of this object has the shape: `(k, nlinks, dim)` where
    `nlinks` is the number of steps taken by the chain and `k` is the number
    of walkers.  Use the `flatchain` property to get the chain flattened to
    `(nlinks, dim)`. For users of older versions, this shape is different so
    be careful!

    """
    def __init__(self, k, *args, **kwargs):
        self.k       = k
        self.a       = kwargs.pop("a", 2.0)
        self.threads = int(kwargs.pop("threads", 1))
        self.pool    = kwargs.pop("pool", None)

        super(EnsembleSampler, self).__init__(*args, **kwargs)
        assert self.k%2 == 0 and self.k >= 2*self.dim

        if self.threads > 1 and self.pool is None:
            self.pool = multiprocessing.Pool(self.threads)

    def reset(self):
        """Clear `chain`, `lnprobability` and the bookkeeping parameters."""
        super(EnsembleSampler, self).reset()
        self.ensembles = [Ensemble(self), Ensemble(self)]
        self.naccepted = np.zeros(self.k)
        self._chain  = np.empty((self.k, 0, self.dim))
        self._lnprob = np.empty((self.k, 0))

    def sample(self, p0, lnprob0=None, rstate0=None, iterations=1, **kwargs):
        """
        Advance the chain iterations steps as an iterator.

        #### Arguments

        * `pos0` (numpy.ndarray): A list of the initial positions of the
          walkers in the parameter space. The shape is `(k, dim)`.

        #### Keyword Arguments

        * `lnprob0` (numpy.ndarray): The list of log posterior probabilities
          for the walkers at positions given by `p0`. If `lnprob is None`,
          the initial values are calculated. The shape is `(k, dim)`.
        * `rstate0` (tuple): The state of the random number generator.
          See the `Sampler.random_state` property for details.
        * `iterations` (int): The number of steps to run. (default: 1)

        #### Yields

        * `pos` (numpy.ndarray): A list of the current positions of the
          walkers in the parameter space. The shape is `(k, dim)`.
        * `lnprob` (numpy.ndarray): The list of log posterior probabilities
          for the walkers at positions given by `pos`. The shape is
          `(k, dim)`.
        * `rstate` (tuple): The state of the random number generator.

        """
        storechain = kwargs.pop("storechain", True)
        thin = kwargs.pop("thin", 1)

        # Try to set the initial value of the random number generator. This
        # fails silently if it doesn't work but that's what we want because
        # we'll just interpret any garbage as letting the generator stay in
        # it's current state.
        self.random_state = rstate0

        # Split the ensemble in half and assign the positions to the two
        # `Ensemble`s.
        p = np.array(p0)
        halfk = int(self.k/2)
        self.ensembles[0].pos = p[:halfk]
        self.ensembles[1].pos = p[halfk:]

        # If the initial log-probabilities were not provided, calculate them
        # now.
        lnprob = lnprob0
        if lnprob is not None:
            self.ensembles[0].lnprob = lnprob[:halfk]
            self.ensembles[1].lnprob = lnprob[halfk:]
        else:
            lnprob = np.zeros(self.k)
            for k, ens in enumerate(self.ensembles):
                ens.lnprob = ens.get_lnprob()
                lnprob[halfk*k:halfk*(k+1)] = ens.lnprob

        # Here, we resize chain in advance for performance. This actually
        # makes a pretty big difference.
        if storechain:
            N = int(iterations/thin)
            self._chain = np.concatenate((self._chain,
                    np.zeros((self.k, N, self.dim))), axis=1)
            self._lnprob = np.concatenate((self._lnprob,
                                           np.zeros((self.k, N))), axis=1)

        i0 = self.iterations
        for i in xrange(int(iterations)):
            self.iterations += 1

            # Loop over the two ensembles, calculating the proposed positions.
            for k, ens in enumerate(self.ensembles):
                q, newlnprob, accept = \
                        self.ensembles[(k+1)%2].propose_position(ens)
                fullaccept = np.zeros(self.k,dtype=bool)
                fullaccept[halfk*k:halfk*(k+1)] = accept

                # Update the `Ensemble`'s walker positions.
                if np.any(accept):
                    lnprob[fullaccept] = newlnprob[accept]
                    p[fullaccept] = q[accept]

                    ens.pos[accept] = q[accept]
                    ens.lnprob[accept] = newlnprob[accept]

                    self.naccepted[fullaccept] += 1

            if storechain and i%thin== 0:
                ind = i0 + int(i/thin)
                self._chain[:,ind,:] = p
                self._lnprob[:,ind]  = lnprob

            # Yield the result as an iterator so that the user can do all
            # sorts of fun stuff with the results so far.
            yield p, lnprob, self.random_state

    @property
    def flatchain(self):
        """
        A shortcut for accessing chain flattened along the zeroth (walker)
        axis.

        """
        s = self.chain.shape
        return self.chain.reshape(s[0]*s[1], s[2])

    @property
    def acor(self):
        """
        The autocorrelation time of each parameter in the chain (length:
        `dim`) as estimated by the `acor` module.

        """
        if acor is None:
            raise ImportError("acor")
        s = self.dim
        t = np.zeros(s)
        for i in range(s):
            t[i] = acor.acor(self.chain[:,:,i].T)[0]
        return t

class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when `args` are
    also included.

    """
    def __init__(self, f, args):
        self.f = f
        self.args = args
    def __call__(self, x):
        return self.f(x, *self.args)

# === Ensemble ===
class Ensemble(object):
    """
    A sub-ensemble object that actually does the heavy lifting of the
    likelihood calculations and proposals of a new position.

    #### Arguments

    * `sampler` (Sampler): The sampler object that this sub-ensemble should
      be connected to.

    """

    def __init__(self, sampler):
        self.sampler = sampler
        # Do a little bit of _magic_ to make the likelihood call with
        # `args` pickleable.
        self.lnprobfn = _function_wrapper(sampler.lnprobfn, sampler.args)

    def get_lnprob(self, pos=None):
        """
        Calculate the vector of log-probability for the walkers.

        #### Keyword Arguments

        * `pos` (numpy.ndarray): The position vector in parameter space where
          the probability should be calculated. This defaults to the current
          position unless a different one is provided.

        #### Returns

        * `lnprob` (numpy.ndarray): A vector of log-probabilities with one
          entry for each walker in this sub-ensemble.

        """
        if pos is None:
            p = self.pos
        else:
            p = pos

        # If the `pool` property of the sampler has been set (i.e. we want
        # to use `multiprocessing`), use the `pool`'s map method. Otherwise,
        # just use the built-in `map` function.
        if self.sampler.pool is not None:
            M = self.sampler.pool.map
        else:
            M = map

        # Calculate the probabilities.
        lnprob = np.array(M(self.lnprobfn, [p[i]
                    for i in range(len(p))]))

        return lnprob

    def propose_position(self, ensemble):
        """
        Propose a new position for another ensemble given the current positions

        #### Arguments

        * `ensemble` (Ensemble): The ensemble to be advanced.

        #### Returns

        * `q` (numpy.array): The new proposed positions for the walkers in
          `ensemble`.
        * `newlnprob` (numpy.ndarray): The vector of log-probabilities at
          the positions given by `q`.
        * `accept` (numpy.ndarray): A vector of `bool`s indicating whether or
          not the proposed position for each walker should be accepted.

        """
        s = np.atleast_2d(ensemble.pos)
        Ns = len(s)
        c = np.atleast_2d(self.pos)
        Nc = len(c)

        # Generate the vectors of random numbers that will produce the
        # proposal.
        zz = ((self.sampler.a - 1.) * self.sampler._random.rand(Ns) + 1)**2.\
                / self.sampler.a
        rint = self.sampler._random.randint(Nc, size=(Ns,))

        # Calculate the proposed positions and the log-probability there.
        q = c[rint] - zz[:,np.newaxis]*(c[rint]-s)
        newlnprob = ensemble.get_lnprob(q)

        # Decide whether or not the proposals should be accepted.
        lnpdiff = (self.sampler.dim - 1.) * np.log(zz) \
                + newlnprob - ensemble.lnprob
        accept = (lnpdiff > np.log(self.sampler._random.rand(len(lnpdiff))))

        return q, newlnprob, accept

