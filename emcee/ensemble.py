# encoding: utf-8
"""
An affine invariant Markov chain Monte Carlo (MCMC) sampler.

Goodman & Weare, Ensemble Samplers With Affine Invariance
   Comm. App. Math. Comp. Sci., Vol. 5 (2010), No. 1, 65–80

"""

__all__ = ['EnsembleSampler', 'MH_proposal_axisaligned']

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
        dangerous = kwargs.pop('live_dangerously', False)

        super(EnsembleSampler, self).__init__(*args, **kwargs)
        self.lnprobfn = _function_wrapper(self.lnprobfn, self.args)
        if not dangerous:
            assert self.k%2 == 0 and self.k >= 2*self.dim

        if self.threads > 1 and self.pool is None:
            self.pool = multiprocessing.Pool(self.threads)

    @staticmethod
    def sampleBall(p0, stdev, nw):
        '''Produce a ball of walkers around an initial parameter value 'p0'
        with axis-aligned standard deviation 'stdev', for 'nw' walkers.'''
        assert(len(p0) == len(stdev))
        return np.vstack([p0 + stdev * np.random.normal(size=len(p0))
                          for i in range(nw)])

    def reset(self):
        """Clear `chain`, `lnprobability` and the bookkeeping parameters."""
        super(EnsembleSampler, self).reset()
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
        mh_proposal = kwargs.pop('mh_proposal', None)

        # Try to set the initial value of the random number generator. This
        # fails silently if it doesn't work but that's what we want because
        # we'll just interpret any garbage as letting the generator stay in
        # its current state.
        self.random_state = rstate0

        # Split the ensemble in half
        p = np.array(p0)
        halfk = int(self.k/2)

        # If the initial log-probabilities were not provided, calculate them
        # now.
        lnprob = lnprob0
        if lnprob is None:
            lnprob = self._getlnprob(p)

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

            # If we were passed a Metropolis-Hastings proposal
            # function, use it.
            if mh_proposal is not None:
                # Draw proposed positions & evaluate lnprob there
                q = mh_proposal(p)
                newlnp = self._getlnprob(q)
                # Accept if newlnp is better; and ...
                acc = (newlnp > lnprob)
                # ... sometimes accept for steps that got worse
                worse = np.flatnonzero(acc == False)
                acc[worse] = ((newlnp[worse] - lnprob[worse]) >
                              np.log(self._random.rand(len(worse))))
                del worse
                lnprob[acc] = newlnp[acc]
                p[acc] = q[acc]
                self.naccepted[acc] += 1

            else:
                # Loop over the two ensembles, calculating the proposed positions.
                # Slices for the first and second halves
                first,second = slice(halfk), slice(halfk, self.k)
                for S0,S1 in [(first,second), (second,first)]:
                    q,newlnp,acc = self._propose_stretch(p[S0], p[S1], lnprob[S0])
                    if np.any(acc):
                        # Update the positions, lnprobs, and acceptance counts
                        lnprob[S0][acc] = newlnp[acc]
                        p[S0][acc] = q[acc]
                        self.naccepted[S0][acc] += 1
                
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
            t[i] = acor.acor(self.chain[:,:,i])[0]
        return t

    def _map(self, func, iterable):
        """ Dispatch to the `pool' for parallel map, or the builtin. """
        if self.pool is not None:
            M = self.pool.map
        else:
            M = map
        return M(func, iterable)

    def _getlnprob(self, p):
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
        return np.array(self._map(self.lnprobfn, p))

    def _propose_stretch(self, p0, p1, lnprob0):
        """
        Propose a new position for one ensemble given another

        #### Arguments

        * `p0`: (numpy array): The positions from which to jump
        * `lnprob0': (numpy array): The ln-probs at p0
        * `p1`: (numpy array): The other ensemble

        #### Returns

        * `q` (numpy.array): The new proposed positions for the walkers in
          `p0`.
        * `newlnprob` (numpy.ndarray): The vector of log-probabilities at
          the positions given by `q`.
        * `accept` (numpy.ndarray): A vector of `bool`s indicating whether or
          not the proposed position for each walker should be accepted.

        """
        s = np.atleast_2d(p0)
        Ns = len(s)
        c = np.atleast_2d(p1)
        Nc = len(c)

        # Generate the vectors of random numbers that will produce the
        # proposal.
        zz = ((self.a - 1.) * self._random.rand(Ns) + 1)**2. / self.a
        rint = self._random.randint(Nc, size=(Ns,))

        # Calculate the proposed positions
        q = c[rint] - zz[:,np.newaxis] * (c[rint] - s)
        # ... and the log-probability there.
        newlnprob = self._getlnprob(q)

        # Decide whether or not the proposals should be accepted.
        lnpdiff = (self.dim - 1.) * np.log(zz) + newlnprob - lnprob0
        accept = (lnpdiff > np.log(self._random.rand(len(lnpdiff))))
        return q, newlnprob, accept
        

class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when `args` are
    also included.

    """
    def __init__(self, f, args):
        self.f = f
        self.args = args
    def __call__(self, x):
        try:
            return self.f(x, *self.args)
        except:
            import traceback
            print 'emcee: Exception while calling your likelihood function:'
            print '  params:', x
            print '  args:', self.args
            print '  exception:'
            traceback.print_exc()
            raise

class MH_proposal_axisaligned(object):
    """
    A Metropolis-Hastings proposal, with axis-aligned Gaussian steps,
    for convenient use as the 'mh_proposal' option to EnsembleSampler.sample.
    """
    def __init__(self, stdev):
        self.stdev = stdev
    def __call__(self, X):
        (nw,npar) = X.shape
        assert(len(self.stdev) == npar)
        return X + self.stdev * np.random.normal(size=X.shape)
