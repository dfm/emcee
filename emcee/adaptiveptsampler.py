#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["AdaptivePTSampler"]

import sys
import numpy as np
import numpy.random as nr
import multiprocessing as multi

from . import autocorr
from .sampler import Sampler
from .ptsampler import PTSampler
from .ptsampler import PTLikePrior

class AdaptivePTState:
    def __init__(self, time, p, betas):
        self.time = time
        self.p = p
        self.betas = betas

class AdaptivePTSampler(PTSampler):
    def __init__(self, *args, ladder_callback=None, evolution_time=100, target_acceptance=0.25, **kwargs):
        super(AdaptivePTSampler, self).__init__(*args, **kwargs)

        self.ladder_callback = ladder_callback
        self.evolution_time = evolution_time
        self.target_acceptance = target_acceptance
        self.nswap_pairs_old = np.zeros(self.ntemps - 1, dtype=np.float)
        self.nswap_pairs_old_accepted = np.zeros(self.ntemps - 1, dtype=np.float)

    def reset(self):
        super(AdaptivePTSampler, self).reset()

        self.nswap_pairs_old = np.zeros(self.ntemps - 1, dtype=np.float)
        self.nswap_pairs_old_accepted = np.zeros(self.ntemps - 1, dtype=np.float)

    def serialize(self):
        """
        Serializes the current sampler object to a string.

        """
        return AdaptivePTState()

    def sample(self, p0, lnprob0=None, lnlike0=None, iterations=1,
            thin=1, storechain=True, evolve_t=True, t0=0):
        """
        Advance the chains ``iterations`` steps as a generator.

        :param p0:
            The initial positions of the walkers.  Shape should be
            ``(ntemps, nwalkers, dim)``.

        :param lnprob0: (optional)
            The initial posterior values for the ensembles.  Shape
            ``(ntemps, nwalkers)``.

        :param lnlike0: (optional)
            The initial likelihood values for the ensembles.  Shape
            ``(ntemps, nwalkers)``.

        :param iterations: (optional)
            The number of iterations to preform.

        :param thin: (optional)
            The number of iterations to perform between saving the
            state to the internal chain.

        :param storechain: (optional)
            If ``True`` store the iterations in the ``chain``
            property.

        :param evolve_t: (optional)
            If ``True``, optimise the temperature ladder over time.

        :param t0: (optional)
            A value with which to initialise the iteration counter;
            only necessary if continuing an old run and evolve_t=True.

        At each iteration, this generator yields

        * ``p``, the current position of the walkers.

        * ``lnprob`` the current posterior values for the walkers.

        * ``lnlike`` the current likelihood values for the walkers.

        """

        # Check temperature ladder for (in)sanity.
        if evolve_t and not np.all(np.diff(self.betas) < 0) and not np.all(np.diff(self.betas) > 0):
            raise ValueError('Temperature ladder must be either ascending or descending.')

        p = np.copy(np.array(p0))
        mapf = map if self.pool is None else self.pool.map
        betas = self._betas.reshape((-1, 1))

        # If we have no lnprob or logls compute them
        if lnprob0 is None or lnlike0 is None:
            fn = PTLikePrior(self.logl, self.logp, self.loglargs,
                             self.logpargs, self.loglkwargs, self.logpkwargs)
            results = list(mapf(fn, p.reshape((-1, self.dim))))

            logls = np.array([r[0] for r in results]).reshape((self.ntemps,
                                                               self.nwalkers))
            logps = np.array([r[1] for r in results]).reshape((self.ntemps,
                                                               self.nwalkers))

            lnlike0 = logls
            lnprob0 = logls * betas + logps

        lnprob = lnprob0
        logl = lnlike0
        logp = logps

        # Expand the chain in advance of the iterations
        if storechain:
            isave = self._update_chain(iterations / thin)

        # Start recording temperatures.
        if evolve_t:
            self._beta_history = np.zeros((self.ntemps, iterations / self.evolution_time))

        for i in range(iterations):
            for j in [0, 1]:
                # Get positions of walkers to be updated and walker to be sampled.
                jupdate = j
                jsample = (j + 1) % 2
                pupdate = p[:, jupdate::2, :]
                psample = p[:, jsample::2, :]

                us = np.random.uniform(size=(self.ntemps, self.nwalkers/2))
                zs = np.square(1.0 + (self.a-1.0)*us)/self.a

                qs = np.zeros((self.ntemps, self.nwalkers/2, self.dim))
                for k in range(self.ntemps):
                    js = np.random.randint(0, high=self.nwalkers / 2,
                                           size=self.nwalkers / 2)
                    qs[k, :, :] = psample[k, js, :] + zs[k, :].reshape(
                        (self.nwalkers / 2, 1)) * (pupdate[k, :, :] -
                                                   psample[k, js, :])

                fn = PTLikePrior(self.logl, self.logp, self.loglargs,
                                 self.logpargs, self.loglkwargs,
                                 self.logpkwargs)
                results = list(mapf(fn, qs.reshape((-1, self.dim))))

                qslogls = np.array([r[0] for r in results]).reshape(
                    (self.ntemps, self.nwalkers/2))
                qslogps = np.array([r[1] for r in results]).reshape(
                    (self.ntemps, self.nwalkers/2))
                qslnprob = qslogls * betas + qslogps

                logpaccept = (self.dim-1)*np.log(zs) + qslnprob \
                    - lnprob[:, jupdate::2]
                logrs = np.log(np.random.uniform(low=0.0, high=1.0,
                                                 size=(self.ntemps,
                                                       self.nwalkers/2)))

                accepts = logrs < logpaccept
                accepts = accepts.flatten()

                pupdate.reshape((-1, self.dim))[accepts, :] = \
                    qs.reshape((-1, self.dim))[accepts, :]
                lnprob[:, jupdate::2].reshape((-1,))[accepts] = \
                    qslnprob.reshape((-1,))[accepts]
                logl[:, jupdate::2].reshape((-1,))[accepts] = \
                    qslogls.reshape((-1,))[accepts]
                logp[:, jupdate::2].reshape((-1,))[accepts] = \
                    qslogps.reshape((-1,))[accepts]

                accepts = accepts.reshape((self.ntemps, self.nwalkers/2))

                self.nprop[:, jupdate::2] += 1.0
                self.nprop_accepted[:, jupdate::2] += accepts

            p, lnprob, logl = self._temperature_swaps(p, lnprob, logl)

            t = i + t0
            if evolve_t and (t + 1) % self.evolution_time == 0:
                dbetas = self._evolve_ladder(t)
                if dbetas != None:
                    self._betas += dbetas
                    betas = self.betas.reshape((-1, 1))
                    lnprob += dbetas.reshape((-1, 1)) * logl

                    # Store the new ladder for reference.
                    self._beta_history[:, t] = self.betas
                    if callable(self.ladder_callback):
                        self.ladder_callback(self)

                self.nswap_pairs_old = self.nswap_pairs.copy()
                self.nswap_pairs_old_accepted = self.nswap_pairs_accepted.copy()

            # Check that posterior is correct.
            values = lnprob - (betas * logl + logp)
            condition = np.abs(values) < 1e-10
            assert condition.all(), \
                    'Posterior does not match likelihood and prior at step {:}: {:}'.format(i, values[np.logical_not(condition)])

            if (i + 1) % thin == 0:
                if storechain:
                    self._chain[:, :, isave, :] = p
                    self._lnprob[:, :, isave, ] = lnprob
                    self._lnlikelihood[:, :, isave] = logl
                    isave += 1

            yield p, lnprob, logl

    def _evolve_ladder(self, t):
        betas = self._betas.copy()
        descending = betas[-1] > betas[0]
        if descending:
            betas = betas[::-1]

        lag = 500 * self.evolution_time
        sigma = 1e2

        # Don't allow chains to move by more than 45% of the log spacing to the adjacent one (to
        # avoid collisions).
        a = 0.45
        kappa0 = a * lag / (t + lag)

        if np.any(self.nswap_pairs - self.nswap_pairs_old < self.evolution_time):
            # Not enough swaps accumulated; abort.
            return None
        As = self.tswap_acceptance_fraction_pairs_recent
        loggammas = -np.diff(np.log(betas))

        kappa = np.zeros(len(betas))
        dlogbetas = np.zeros(len(betas))
        top = len(betas) - 2 # The index of the topmost chain subject to the "normal" dynamics.
        if self.target_acceptance != None:
            # Drive the chains to a specified acceptance ratio.
            A0 = self.target_acceptance

            # Compute new temperature spacings from acceptance spacings.
            As = np.concatenate(([ A0 ], As))
            dlogbetas = -(As - A0)

            # Topmost chain shouldn't change by more than the gap between it and the next lowest.
            kappa[-1] = loggammas[-1]
        else:
            # Allow the chains to equilibrate to even acceptance-spacing for all chains. Topmost
            # chains aim for this acceptance ratio:
            A0 = 0.99
            top -= 1

            # Drive chains 1 to N-2 toward even spacing.
            dlogbetas[1:-2] = -(As[:-2] - As[1:-1])

            # Drive second-topmost chain until it reaches specified acceptance with topmost
            # chain. Topmost chain is dealt with later.
            kappa[-2] = loggammas[-2]
            dlogbetas[-2] = -(A0 - As[-1])

        # Calculate dynamics time-scale (kappa). Limit the adjustment of log(beta) to less than half
        # the size of the gap in the direction in which the chain is moving (to avoid the chains
        # bouncing off each other). If log(beta) is decreasing, chain is ascending, so use gap with
        # next-highest chain (and vice versa).
        kappa[1:top + 1] = np.select([ -dlogbetas[1:top + 1] < 0, -dlogbetas[1:top + 1] > 0 ],
                                     [ loggammas[:top], loggammas[1:top + 1] ])
        kappa *= kappa0

        # Compute new temperature spacings.
        self.kappa = kappa
        dlogbetas *= kappa
        loggammas -= np.diff(dlogbetas)

        # Ensure log-spacings are positive and adjust temperature chain. Whereever a negative
        # spacing is replaced by zero, must compensate by increasing subsequent spacing in order to
        # preserve higher temperatures.
        if self.target_acceptance == None:
            # Top two chains are exempt from spacing preservation, since their spacing is fixed by
            # construction.
            gaps = -np.concatenate((np.minimum(loggammas, 0)[:-2], [0, 0]))
        else:
            gaps = -np.concatenate((np.minimum(loggammas, 0)[:-1], [0]))
        loggammas = np.maximum(loggammas, 0) + np.roll(gaps, 1)

        if self.target_acceptance == None:
            # Now fix the top spacing at much more than whatever the next one down is, to ensure
            # good prior sampling.
            loggammas[-1] = np.log(sigma) + loggammas[-2]

        # Finally, compute the new ladder.
        betas[1:] = np.exp(-np.cumsum(loggammas))

        # Un-reverse the ladder if need be.
        if descending:
            betas = betas[::-1]

        # Don't mutate the ladder here; let the client code do that.
        return betas - self._betas

    def _update_chain(self, nsave):
        if self._chain is None:
            isave = 0
            self._chain = np.zeros((self.ntemps, self.nwalkers, nsave,
                                    self.dim))
            self._lnprob = np.zeros((self.ntemps, self.nwalkers, nsave))
            self._lnlikelihood = np.zeros((self.ntemps, self.nwalkers,
                                           nsave))
        else:
            isave = self._chain.shape[2]
            self._chain = np.concatenate((self._chain,
                                          np.zeros((self.ntemps,
                                                    self.nwalkers,
                                                    nsave, self.dim))),
                                         axis=2)
            self._lnprob = np.concatenate((self._lnprob,
                                           np.zeros((self.ntemps,
                                                     self.nwalkers,
                                                     nsave))),
                                          axis=2)
            self._lnlikelihood = np.concatenate((self._lnlikelihood,
                                                 np.zeros((self.ntemps,
                                                           self.nwalkers,
                                                           nsave))),
                                                axis=2)

        return isave

    @property
    def tswap_acceptance_fraction_pairs_recent(self):
        """
        Returns an array of recently accepted temperature swap fractions for
        each pair of temperatures; shape ``(ntemps - 1, )``. If no swaps have
        been accumulated in at least one chain, returns ``None``.

        """
        accepted = self.nswap_pairs_accepted - self.nswap_pairs_old_accepted
        swaps =  self.nswap_pairs - self.nswap_pairs_old
        return accepted / swaps if np.all(swaps > 0) else None
