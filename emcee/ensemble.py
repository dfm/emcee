# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["EnsembleSampler"]

import logging
from collections import Iterable

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(f, *args, **kwargs):
        logging.warn("You must install the tqdm library to use progress "
                     "indicators with emcee")
        return f

from .backends import Backend
from .moves import StretchMove
from .utils import deprecated, deprecation_warning


class EnsembleSampler(object):
    """An ensemble MCMC sampler

    Args:
        nwalkers (int): The number of walkers in the ensemble.
        ndim (int): Number of dimensions in the parameter space.
        log_prob_fn (callable): A function that takes a vector in the
            parameter space as input and returns the natural logarithm of the
            posterior probability (up to an additive constant) for that
            position.
        moves (Optional): This can be a single move object, a list of moves,
            or a "weighted" list of the form ``[(emcee3.moves.StretchMove(),
            0.1), ...]``. When running, the sampler will randomly select a
            move from this list (optionally with weights) for each proposal.
            (default: :class:`StretchMove`)
        args (Optional): A list of extra positional arguments for
            ``log_prob_fn``. ``log_prob_fn`` will be called with the sequence
            ``log_pprob_fn(p, *args, **kwargs)``.
        kwargs (Optional): A dict of extra keyword arguments for
            ``log_prob_fn``. ``log_prob_fn`` will be called with the sequence
            ``log_pprob_fn(p, *args, **kwargs)``.
        pool (Optional): An object with a ``map`` method that follows the same
            calling sequence as the built-in ``map`` function. This is
            generally used to compute the log-probabilities for the ensemble
            in parallel.
        backend (Optional): Either a :class:`backends.Backend` or a subclass
            (like :class:`backends.HDFBackend) that is used to store and
            serialize the state of the chain. By default, the chain is stored
            as a set of numpy arrays in memory, but new backends can be
            written to support other mediums.
        vectorize (Optional[bool]): If ``True``, ``log_prob_fn`` is expected
            to accept a list of position vectors instead of just one.
            (default: ``False``)

    """
    def __init__(self, nwalkers, ndim, log_prob_fn,
                 pool=None, moves=None,
                 args=None, kwargs=None,
                 backend=None,
                 vectorize=False,
                 # Deprecated...
                 a=None, postargs=None, threads=None,  live_dangerously=None,
                 runtime_sortingfn=None):
        # Warn about deprecated arguments
        if a is not None:
            deprecation_warning(
                "The 'a' argument is deprecated, use 'moves' instead")
        if threads is not None:
            deprecation_warning(
                "The 'threads' argument is deprecated")
        if runtime_sortingfn is not None:
            deprecation_warning(
                "The 'runtime_sortingfn' argument is deprecated")
        if live_dangerously is not None:
            deprecation_warning(
                "The 'live_dangerously' argument is deprecated")

        # Parse the move schedule
        if moves is None:
            self._moves = [StretchMove()]
            self._weights = [1.0]
        elif isinstance(moves, Iterable):
            try:
                self._moves, self._weights = zip(*moves)
            except TypeError:
                self._moves = moves
                self._weights = np.ones(len(moves))
        else:
            self._moves = [moves]
            self._weights = [1.0]
        self._weights = np.atleast_1d(self._weights).astype(float)
        self._weights /= np.sum(self._weights)

        self.pool = pool
        self.vectorize = vectorize

        self.ndim = ndim
        self.nwalkers = nwalkers
        self.backend = Backend() if backend is None else backend

        # Deal with re-used backends
        if not self.backend.initialized:
            self.reset()
            state = np.random.get_state()
        else:
            # Check the backend shape
            if self.backend.shape != (self.nwalkers, self.ndim):
                raise ValueError(("the shape of the backend ({0}) is "
                                  "incompatible with the shape of the sampler "
                                  "({1})").format(
                                      self.backend.shape,
                                      (self.nwalkers, self.ndim)
                                  ))

            # Get the last random state
            state = self.backend.random_state
            if state is None:
                state = np.random.get_state()

            # Grab the last step so that we can restart
            it = self.backend.iteration
            if it > 0:
                last = [self.backend.get_chain(discard=it-1)[0]]
                last += [self.backend.get_log_prob(discard=it-1)[0]]
                last += [state]
                blob = self.backend.get_blobs(discard=it-1)
                if blob is not None:
                    last += [blob[0]]
                self._last_run_mcmc_result = tuple(last)

        # This is a random number generator that we can easily set the state
        # of without affecting the numpy-wide generator
        self._random = np.random.mtrand.RandomState()
        self._random.set_state(state)

        # Do a little bit of _magic_ to make the likelihood call with
        # ``args`` and ``kwargs`` pickleable.
        self.log_prob_fn = _function_wrapper(log_prob_fn, args, kwargs)

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

    @random_state.setter  # NOQA
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
    def iteration(self):
        return self.backend.iteration

    def reset(self):
        """
        Reset the bookkeeping parameters

        """
        self._last_run_mcmc_result = None
        self.backend.reset(self.nwalkers, self.ndim)

    def __getstate__(self):
        # In order to be generally picklable, we need to discard the pool
        # object before trying.
        d = self.__dict__
        d.pop("pool", None)
        return d

    def sample(self, p0, log_prob0=None, rstate0=None, blobs0=None,
               iterations=1, thin_by=1, thin=None, store=True, progress=False):
        """Advance the chain as a generator

        Args:
            p0 (ndarray[nwalkers, ndim]): The initial positions of the walkers
                in the parameter space.
            log_prob0 (Optional[ndarray[nwalkers]]): The log posterior
                probabilities for the walkers at ``p0``. If ``log_prob0 is
                None``, the initial values are calculated.
            rstate0 (Optional): The state of the random number generator.
                See the :attr:`EnsembleSampler.random_state` property for
                details.
            iterations (Optional[int]): The number of steps to run.
            thin_by (Optional[int]): If you only want to store and yield every
                ``thin`` samples in the chain, set thin to an integer greater
                than 1.
            store (Optional[bool]): By default, the sampler stores (in memory)
                the positions and log-probabilities of the samples in the
                chain. If you are using another method to store the samples to
                a file or if you don't need to analyze the samples after the
                fact (for burn-in for example) set ``store`` to ``False``.

        At each iteration, this generator yields:

        * ``pos`` - A list of the current positions of the walkers in the
          parameter space. The shape of this object will be
          ``(nwalkers, dim)``.

        * ``log_prob`` - The list of log posterior probabilities for the
          walkers at positions given by ``pos`` . The shape of this object
          is ``(nwalkers,)``.

        * ``rstate`` - The current state of the random number generator.

        * ``blobs`` - (optional) The metadata "blobs" associated with the
          current position. The value is only returned if ``log_prob_fn``
          returns blobs too.

        """
        # Try to set the initial value of the random number generator. This
        # fails silently if it doesn't work but that's what we want because
        # we'll just interpret any garbage as letting the generator stay in
        # it's current state.
        self.random_state = rstate0
        p = np.array(p0)
        if np.shape(p) != (self.nwalkers, self.ndim):
            raise ValueError("incompatible input dimensions")

        # If the initial log-probabilities were not provided, calculate them
        # now.
        log_prob = log_prob0
        blobs = blobs0
        if log_prob is None:
            log_prob, blobs = self.compute_log_prob(p)
        if np.shape(log_prob) != (self.nwalkers, ):
            raise ValueError("incompatible input dimensions")

        # Check to make sure that the probability function didn't return
        # ``np.nan``.
        if np.any(np.isnan(log_prob)):
            raise ValueError("The initial log_prob was NaN")

        # Deal with deprecated thin argument
        if thin is not None:
            deprecation_warning("The 'thin' argument is deprecated. "
                                "Use 'thin_by' instead.")

            # Check that the thin keyword is reasonable.
            thin = int(thin)
            if thin <= 0:
                raise ValueError("Invalid thinning argument")

            yield_step = 1
            checkpoint_step = thin
            iterations = int(iterations)
            if store:
                N = iterations // checkpoint_step
                self.backend.grow(N, blobs)

        else:
            # Check that the thin keyword is reasonable.
            thin_by = int(thin_by)
            if thin_by <= 0:
                raise ValueError("Invalid thinning argument")

            yield_step = thin_by
            checkpoint_step = thin_by
            iterations = int(iterations)
            if store:
                self.backend.grow(iterations, blobs)

        # Inject the progress bar
        total = int(iterations)
        if progress:
            gen = tqdm(range(total), total=total)
        else:
            gen = range(total)

        i = 0
        for _ in gen:
            for _ in range(yield_step):
                # Choose a random move
                move = self._random.choice(self._moves, p=self._weights)

                # Propose
                p, log_prob, blobs, accepted = move.propose(
                    p, log_prob, blobs, self.compute_log_prob, self._random)

                # Save the new step
                if store and (i + 1) % checkpoint_step == 0:
                    self.backend.save_step(p, log_prob, blobs, accepted,
                                           self.random_state)

                i += 1

            # Yield the result as an iterator so that the user can do all
            # sorts of fun stuff with the results so far.
            if blobs is not None:
                # This is a bit of a hack to keep things backwards
                # compatible.
                yield p, log_prob, self.random_state, blobs
            else:
                yield p, log_prob, self.random_state

    def run_mcmc(self, pos0, N, rstate0=None, log_prob0=None, blobs0=None,
                 **kwargs):
        """
        Iterate :func:`sample` for ``N`` iterations and return the result

        Args:
            pos0: The initial position vector. Can also be ``None`` to resume
                from where :func:``run_mcmc`` left off the last time it
                executed.
            N: The number of steps to run.
            log_prob0 (Optional[ndarray[nwalkers]]): The log posterior
                probabilities for the walkers at ``p0``. If ``log_prob0 is
                None``, the initial values are calculated.
            rstate0 (Optional): The state of the random number generator.
                See the :attr:`EnsembleSampler.random_state` property for
                details.

        Other parameters are directly passed to :func:`sample`.

        This method returns the most recent result from :func:`sample`. The
        particular values vary from sampler to sampler, but the result is
        generally a tuple ``(pos, log_prob, rstate)`` or
        ``(pos, log_prob, rstate, blobs)`` where ``pos`` is the most recent
        position vector (or ensemble thereof), ``log_prob`` is the most recent
        log posterior probability (or ensemble thereof), ``rstate`` is the
        state of the random number generator, and the optional ``blobs`` are
        user-provided large data blobs.

        """
        if pos0 is None:
            if self._last_run_mcmc_result is None:
                raise ValueError("Cannot have pos0=None if run_mcmc has never "
                                 "been called.")
            pos0, log_prob0, rstate0 = self._last_run_mcmc_result[:3]
            if len(self._last_run_mcmc_result) > 3:
                blobs0 = self._last_run_mcmc_result[3]

        for results in self.sample(pos0, log_prob0, rstate0=rstate0,
                                   blobs0=blobs0, iterations=N, **kwargs):
            pass

        # Store so that the ``pos0=None`` case will work
        self._last_run_mcmc_result = results

        return results

    def compute_log_prob(self, coords=None):
        """Calculate the vector of log-probability for the walkers

        Args:
            pos: (Optional[ndarray[..., ndim]]) The position vector in
                parameter space where the probability should be calculated.
                This defaults to the current position unless a different one
                is provided.

        This method returns:

        * log_prob: A vector of log-probabilities with one entry for each
          walker in this sub-ensemble.
        * blob: The list of meta data returned by the ``log_post_fn`` at
          this position or ``None`` if nothing was returned.

        """
        p = coords

        # Check that the parameters are in physical ranges.
        if np.any(np.isinf(p)):
            raise ValueError("At least one parameter value was infinite")
        if np.any(np.isnan(p)):
            raise ValueError("At least one parameter value was NaN")

        # If the `pool` property of the sampler has been set (i.e. we want
        # to use `multiprocessing`), use the `pool`'s map method. Otherwise,
        # just use the built-in `map` function.
        if self.pool is not None:
            M = self.pool.map
        else:
            M = map

        # Run the log-probability calculations (optionally in parallel).
        if self.vectorize:
            results = self.log_prob_fn(p)
        else:
            results = list(M(self.log_prob_fn, [p[i] for i in range(len(p))]))

        try:
            log_prob = np.array([float(l[0]) for l in results])
            blob = [l[1:] for l in results]
            blob = np.array(blob, dtype=np.atleast_1d(blob[0]).dtype)

            # Deal with single blobs in a better way
            if blob.shape[1] == 1:
                m = [slice(None) for i in range(len(blob.shape))]
                m[1] = 0
                blob = blob[m]
        except (IndexError, TypeError):
            log_prob = np.array([float(l) for l in results])
            blob = None

        # Check for log_prob returning NaN.
        if np.any(np.isnan(log_prob)):
            raise ValueError("Probability function returned NaN")

        return log_prob, blob

    @property
    def acceptance_fraction(self):
        """The fraction of proposed steps that were accepted"""
        return self.backend.accepted / float(self.backend.iteration)

    @property
    @deprecated("get_chain()")
    def chain(self):
        chain = self.get_chain()
        return np.swapaxes(chain, 0, 1)

    @property
    @deprecated("get_chain(flat=True)")
    def flatchain(self):
        return self.get_chain(flat=True)

    @property
    @deprecated("get_log_prob()")
    def lnprobability(self):
        return self.get_log_prob()

    @property
    @deprecated("get_log_prob(flat=True)")
    def flatlnprobability(self):
        return self.get_log_prob(flat=True)

    @property
    @deprecated("get_blobs()")
    def blobs(self):
        return self.get_blobs()

    @property
    @deprecated("get_blobs(flat=True)")
    def flatblobs(self):
        return self.get_blobs(flat=True)

    def get_chain(self, **kwargs):
        return self.get_value("chain", **kwargs)
    get_chain.__doc__ = Backend.get_chain.__doc__

    def get_blobs(self, **kwargs):
        return self.get_value("blobs", **kwargs)
    get_blobs.__doc__ = Backend.get_blobs.__doc__

    def get_log_prob(self, **kwargs):
        return self.get_value("log_prob", **kwargs)
    get_log_prob.__doc__ = Backend.get_log_prob.__doc__

    def get_value(self, name, **kwargs):
        return self.backend.get_value(name, **kwargs)

    @property
    @deprecated("get_autocorr_time")
    def acor(self):
        return self.get_autocorr_time()

    def get_autocorr_time(self, **kwargs):
        return self.backend.get_autocorr_time(**kwargs)
    get_autocorr_time.__doc__ = Backend.get_autocorr_time.__doc__


class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.

    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            print("emcee: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise
