# encoding: utf-8
"""
An affine invariant Markov chain Monte Carlo (MCMC) sampler.

Goodman & Weare, Ensemble Samplers With Affine Invariance
   Comm. App. Math. Comp. Sci., Vol. 5 (2010), No. 1, 65–80

"""

from __future__ import print_function

__all__ = ["EnsembleSampler", "Ensemble"]

import multiprocessing
import numpy as np

try:
    import acor
    acor = acor
except ImportError:
    acor = None

from .sampler import Sampler


class EnsembleSampler(Sampler):
    """
    A generalized Ensemble sampler that uses 2 ensembles for parallelization.
    The ``__init__`` function will raise an ``AssertionError`` if
    ``k < 2 * dim`` (and you haven't set the ``live_dangerously`` parameter)
    or if ``k`` is odd.

    **Warning**: The :attr:`chain` member of this object has the shape:
    ``(nwalkers, nlinks, dim)`` where ``nlinks`` is the number of steps
    taken by the chain and ``k`` is the number of walkers.  Use the
    :attr:`flatchain` property to get the chain flattened to
    ``(nlinks, dim)``. For users of pre-1.0 versions, this shape is
    different so be careful!

    :param nwalkers:
        The number of Goodman & Weare "walkers".

    :param dim:
        Number of dimensions in the parameter space.

    :param lnpostfn:
        A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the posterior probability for that
        position.

    :param a: (optional)
        The proposal scale parameter. (default: ``2.0``)

    :param args: (optional)
        A list of extra arguments for ``lnpostfn``. ``lnpostfn`` will be
        called with the sequence ``lnpostfn(p, *args)``.

    :param postargs: (optional)
        Alias of ``args`` for backwards compatibility.

    :param threads: (optional)
        The number of threads to use for parallelization. If ``threads == 1``,
        then the ``multiprocessing`` module is not used but if
        ``threads > 1``, then a ``Pool`` object is created and calls to
        ``lnpostfn`` are run in parallel.

    :param pool: (optional)
        An alternative method of using the parallelized algorithm. If
        provided, the value of ``threads`` is ignored and the
        object provided by ``pool`` is used for all parallelization. It
        can be any object with a ``map`` method that follows the same
        calling sequence as the built-in ``map`` function.

    """
    def __init__(self, nwalkers, dim, lnpostfn, a=2.0, args=[], postargs=None,
            threads=1, pool=None, live_dangerously=False):
        self.k = nwalkers
        self.a = a
        self.threads = threads
        self.pool = pool

        if postargs is not None:
            args = postargs
        super(EnsembleSampler, self).__init__(dim, lnpostfn, args=args)

        assert self.k % 2 == 0, "The number of walkers must be even."
        if not live_dangerously:
            assert self.k >= 2 * self.dim, (
                    "The number of walkers needs to be more than twice the "
                    + "dimension of your parameter space... unless you're "
                    + "crazy!")

        if self.threads > 1 and self.pool is None:
            self.pool = multiprocessing.Pool(self.threads)

    def reset(self):
        """
        Clear the ``chain`` and ``lnprobability`` array. Also reset the
        bookkeeping parameters.

        """
        super(EnsembleSampler, self).reset()
        self.ensembles = [Ensemble(self), Ensemble(self)]
        self.naccepted = np.zeros(self.k)
        self._chain = np.empty((self.k, 0, self.dim))
        self._lnprob = np.empty((self.k, 0))

        # Initialize lists for storing optional metadata blobs.
        self._blobs = [[] for i in range(self.k)]

    def sample(self, p0, lnprob0=None, rstate0=None, blobs0=None,
            iterations=1, thin=1, storechain=True):
        """
        Advance the chain iterations steps as a generator.

        :param p0:
            A list of the initial positions of the walkers in the
            parameter space. It should have the shape ``(nwalkers, dim)``.

        :param lnprob0: (optional)
            The list of log posterior probabilities for the walkers at
            positions given by ``p0``. If ``lnprob is None``, the initial
            values are calculated. It should have the shape ``(k, dim)``.

        :param rstate0: (optional)
            The state of the random number generator.
            See the :attr:`Sampler.random_state` property for details.

        :param iterations: (optional)
            The number of steps to run.

        At each iteration, this generator yields:

        * ``pos`` — A list of the current positions of the walkers in the
          parameter space. The shape of this object will be
          ``(nwalkers, dim)``.

        * ``lnprob`` — The list of log posterior probabilities for the
          walkers at positions given by ``pos``. The shape of this object
          is ``(nwalkers, dim)``.

        * ``rstate`` — The current state of the random number generator.

        * ``blobs`` — (optional) The metadata "blobs" associated with the
          current position. The value is only returned if ``lnpostfn``
          returns blobs too.

        """
        # Try to set the initial value of the random number generator. This
        # fails silently if it doesn't work but that's what we want because
        # we'll just interpret any garbage as letting the generator stay in
        # it's current state.
        self.random_state = rstate0

        # Split the ensemble in half and assign the positions to the two
        # `Ensemble`s.
        p = np.array(p0)
        halfk = int(self.k / 2)
        self.ensembles[0].pos = p[:halfk]
        self.ensembles[1].pos = p[halfk:]

        # If the initial log-probabilities were not provided, calculate them
        # now.
        blobs = []
        lnprob = lnprob0
        if lnprob is not None:
            self.ensembles[0].lnprob = lnprob[:halfk]
            self.ensembles[1].lnprob = lnprob[halfk:]
        else:
            lnprob = np.zeros(self.k)
            for k, ens in enumerate(self.ensembles):
                ens.lnprob, blob = ens.get_lnprob()
                lnprob[halfk * k:halfk * (k + 1)] = ens.lnprob
                if blob is not None:
                    blobs += blob

        # Here, we resize chain in advance for performance. This actually
        # makes a pretty big difference.
        if storechain:
            N = int(iterations / thin)
            self._chain = np.concatenate((self._chain,
                    np.zeros((self.k, N, self.dim))), axis=1)
            self._lnprob = np.concatenate((self._lnprob,
                                           np.zeros((self.k, N))), axis=1)

        i0 = self.iterations
        # Use range instead of xrange for compatability with python 3
        # It is slightly less efficient, but for a realistic number of
        # walkers it isn't too bad
        for i in range(int(iterations)):
            self.iterations += 1

            # Loop over the two ensembles, calculating the proposed positions.
            for k, ens in enumerate(self.ensembles):
                q, newlnprob, accept, blob = \
                        self.ensembles[(k + 1) % 2].propose_position(ens)
                fullaccept = np.zeros(self.k, dtype=bool)
                fullaccept[halfk * k:halfk * (k + 1)] = accept

                # Update the `Ensemble`'s walker positions.
                if np.any(accept):
                    lnprob[fullaccept] = newlnprob[accept]
                    p[fullaccept] = q[accept]

                    ens.pos[accept] = q[accept]
                    ens.lnprob[accept] = newlnprob[accept]

                    self.naccepted[fullaccept] += 1

                    if blob is not None:
                        assert blobs is not None, ("If you start sampling "
                                + "with a given lnprob, you also need to "
                                + "provide the current list of blobs at that "
                                + "position.")
                        ind = np.arange(len(accept))[accept]
                        indfull = np.arange(len(fullaccept))[fullaccept]
                        for j in range(len(ind)):
                            blobs[indfull[j]] = blob[ind[j]]

            if storechain and i % thin == 0:
                ind = i0 + int(i / thin)
                self._chain[:, ind, :] = p
                self._lnprob[:, ind] = lnprob
                if blobs is not None:
                    self._blobs.append(blobs)

            # Yield the result as an iterator so that the user can do all
            # sorts of fun stuff with the results so far.
            if len(self._blobs) > 0:
                # This is a bit of a hack to keep things backwards compatible.
                yield p, lnprob, self.random_state, blobs
            else:
                yield p, lnprob, self.random_state

    @property
    def blobs(self):
        """
        Get the list of "blobs" produced by sampling. The result is a list
        (of length ``iterations``) of ``list``s (of length ``nwalkers``) of
        arbitrary objects. **Note**: this will actually be an empty list if
        your ``lnpostfn`` doesn't return any metadata.

        """
        return self._blobs

    @property
    def flatchain(self):
        """
        A shortcut for accessing chain flattened along the zeroth (walker)
        axis.

        """
        s = self.chain.shape
        return self.chain.reshape(s[0] * s[1], s[2])

    @property
    def acor(self):
        """
        The autocorrelation time of each parameter in the chain (length:
        ``dim``) as estimated by the ``acor`` module.

        """
        if acor is None:
            raise ImportError("acor")
        s = self.dim
        t = np.zeros(s)
        for i in range(s):
            t[i] = acor.acor(self.chain[:, :, i])[0]
        return t


class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    are also included.

    """
    def __init__(self, f, args):
        self.f = f
        self.args = args

    def __call__(self, x):
        try:
            return self.f(x, *self.args)
        except:
            import traceback
            print("emcee: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  exception:")
            traceback.print_exc()
            raise


class Ensemble(object):
    """
    A sub-ensemble object that actually does the heavy lifting of the
    likelihood calculations and proposals of a new position.

    :param sampler:
        The sampler object that this sub-ensemble should be connected to.

    """

    def __init__(self, sampler):
        self.sampler = sampler
        # Do a little bit of _magic_ to make the likelihood call with
        # `args` pickleable.
        self.lnprobfn = _function_wrapper(sampler.lnprobfn, sampler.args)

    def get_lnprob(self, pos=None):
        """
        Calculate the vector of log-probability for the walkers.

        :param pos: (optional)
            The position vector in parameter space where the probability
            should be calculated. This defaults to the current position
            unless a different one is provided.

        This method returns:

        * ``lnprob`` — A vector of log-probabilities with one entry for each
          walker in this sub-ensemble.

        * ``blob`` — The list of meta data returned by the ``lnpostfn`` at
          this position or ``None`` if nothing was returned.

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

        # Run the log-probability calculations (optionally in parallel).
        results = list(M(self.lnprobfn, [p[i] for i in range(len(p))]))

        try:
            lnprob = np.array([l[0] for l in results])
            blob = [l[1] for l in results]
        except (IndexError, TypeError):
            lnprob = np.array(results)
            blob = None

        return lnprob, blob

    def propose_position(self, ensemble):
        """
        Propose a new position for *a different* ensemble given the current
        positions in *this*.

        :param ensemble:
            The ensemble to be advanced.

        This method returns:

        * ``q`` — The new proposed positions for the walkers in ``ensemble``.

        * ``newlnprob`` — The vector of log-probabilities at the positions
          given by ``q``.

        * ``accept`` — A vector of type ``bool`` indicating whether or not
          the proposed position for each walker should be accepted.

        * ``blob`` — The new meta data blobs or ``None`` if nothing was
          returned by ``lnprobfn``.

        """
        s = np.atleast_2d(ensemble.pos)
        Ns = len(s)
        c = np.atleast_2d(self.pos)
        Nc = len(c)

        # Generate the vectors of random numbers that will produce the
        # proposal.
        zz = ((self.sampler.a - 1.) * self.sampler._random.rand(Ns) + 1) ** 2.\
                / self.sampler.a
        rint = self.sampler._random.randint(Nc, size=(Ns,))

        # Calculate the proposed positions and the log-probability there.
        q = c[rint] - zz[:, np.newaxis] * (c[rint] - s)
        newlnprob, blob = ensemble.get_lnprob(q)

        # Decide whether or not the proposals should be accepted.
        lnpdiff = (self.sampler.dim - 1.) * np.log(zz) \
                + newlnprob - ensemble.lnprob
        accept = (lnpdiff > np.log(self.sampler._random.rand(len(lnpdiff))))

        return q, newlnprob, accept, blob
