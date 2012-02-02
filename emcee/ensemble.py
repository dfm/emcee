# encoding: utf-8
"""
An affine invariant Markov chain Monte Carlo (MCMC) sampler.

Goodman & Weare, Ensemble Samplers With Affine Invariance
   Comm. App. Math. Comp. Sci., Vol. 5 (2010), No. 1, 65–80

"""

__all__ = ['EnsembleSampler']

import multiprocessing
import pickle

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

    This is a subclass of the `Sampler` object. See [[sampler.py#sampler]]
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
        if self.pool is not None:
            if len(self.args) > 0:
                self.lnprobfn = _function_wrapper(self.lnprobfn, self.args)
            try:
                pickle.dumps(self.lnprobfn)
            except pickle.PicklingError:
                print "Warning: lnprobfn is not picklable. "\
                        "Parallelization probably won't work"

    def reset(self):
        """Clear `chain`, `lnprobability` and the bookkeeping parameters."""
        super(EnsembleSampler, self).reset()
        self.ensembles = [Ensemble(self), Ensemble(self)]
        self.naccepted = np.zeros(self.k)
        self._chain  = np.empty((self.k, 0, self.dim))
        self._lnprob = np.empty((self.k, 0))

    def sample(self, p0, lnprob0=None, rstate0=None, storechain=True,
            resample=1, iterations=1):
        """
        Advances the chain iterations steps as an iterator

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
        self.random_state = rstate0

        p = np.array(p0)
        halfk = int(self.k/2)
        lnprob = lnprob0
        self.ensembles[0].pos = p[:halfk]
        self.ensembles[1].pos = p[halfk:]
        if lnprob is not None:
            self.ensembles[0].lnprob = lnprob[:halfk]
            self.ensembles[1].lnprob = lnprob[halfk:]
        else:
            lnprob = np.zeros(self.k)
            for k, ens in enumerate(self.ensembles):
                ens.lnprob = ens.get_lnprob()
                lnprob[halfk*k:halfk*(k+1)] = ens.lnprob

        # resize chain
        if storechain:
            N = int(iterations/resample)
            self._chain = np.concatenate((self._chain,
                    np.zeros((self.k, N, self.dim))), axis=1)
            self._lnprob = np.concatenate((self._lnprob, np.zeros((self.k, N))),
                    axis=1)

        i0 = self.iterations
        for i in xrange(int(iterations)):
            self.iterations += 1

            for k, ens in enumerate(self.ensembles):
                q, newlnprob, accept = self.ensembles[(k+1)%2].propose_position(ens)
                fullaccept = np.zeros(self.k,dtype=bool)
                fullaccept[halfk*k:halfk*(k+1)] = accept
                if np.any(accept):
                    lnprob[fullaccept] = newlnprob[accept]
                    p[fullaccept] = q[accept]

                    ens.pos[accept] = q[accept]
                    ens.lnprob[accept] = newlnprob[accept]

                    self.naccepted[fullaccept] += 1

            if storechain and i%resample == 0:
                ind = i0 + int(i/resample)
                self._chain[:,ind,:] = p
                self._lnprob[:,ind]  = lnprob

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
        The autocorrelation time of each parameter in the chain (length: `dim`)
        as estimated by the `acor` module.

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
    also included

    """
    def __init__(self, f, args):
        self.f = f
        self.args = args
    def __call__(self, x):
        return self.f(x, *self.args)

# === Ensemble ===
class Ensemble(object):
    def __init__(self, sampler):
        self._sampler = sampler

    def get_lnprob(self, pos=None):
        if pos is None:
            p = self.pos
        else:
            p = pos

        if self._sampler.pool is not None:
            M = self._sampler.pool.map
        else:
            M = map
        lnprob = np.array(M(self._sampler.get_lnprob, [p[i]
                    for i in range(len(p))]))

        return lnprob

    def propose_position(self, ensemble):
        """
        Propose a new position for another ensemble given the current positions

        #### Parameters

        * `ensemble` (Ensemble): The ensemble to be advanced.

        """
        s = np.atleast_2d(ensemble.pos)
        Ns = len(s)
        c = np.atleast_2d(self.pos)
        Nc = len(c)

        zz = ((self._sampler.a-1.)*self._sampler._random.rand(Ns)+1)**2./self._sampler.a
        rint = self._sampler._random.randint(Nc, size=(Ns,))

        # propose new walker position and calculate the lnprobability
        q = c[rint] - zz[:,np.newaxis]*(c[rint]-s)
        newlnprob = ensemble.get_lnprob(q)

        lnpdiff = (self._sampler.dim - 1.) * np.log(zz) + newlnprob - ensemble.lnprob
        accept = (lnpdiff > np.log(self._sampler._random.rand(len(lnpdiff))))

        return q, newlnprob, accept

