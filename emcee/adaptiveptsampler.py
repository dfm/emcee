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

class AdaptivePTSampler(PTSampler):
    def __init__(self, *args, ladder_callback=None, evolution_time=100, forcing_constant=20, target_acceptance=0.25, **kwargs):
        super(AdaptivePTSampler, self).__init__(*args, **kwargs)

        self.ladder_callback = ladder_callback
        self.evolution_time = evolution_time
        self.forcing_constant = forcing_constant
        self.target_acceptance = target_acceptance
        self.nswap_between_old = np.zeros(self.ntemps - 1, dtype=np.float)
        self.nswap_between_old_accepted = np.zeros(self.ntemps - 1, dtype=np.float)

    def reset(self):
        super(AdaptivePTSampler, self).reset()

        self.nswap_between_old = np.zeros(self.ntemps - 1, dtype=np.float)
        self.nswap_between_old_accepted = np.zeros(self.ntemps - 1, dtype=np.float)

    def sample(self, p0, lnprob0=None, lnlike0=None, iterations=1,
            thin=1, storechain=True, evolve_t=True):
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

        At each iteration, this generator yields

        * ``p``, the current position of the walkers.

        * ``lnprob`` the current posterior values for the walkers.

        * ``lnlike`` the current likelihood values for the walkers.

        """
        p = np.copy(np.array(p0))
        mapf = map if self.pool is None else self.pool.map

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
            lnprob0 = logls * self.betas.reshape((self.ntemps, 1)) + logps

        lnprob = lnprob0
        logl = lnlike0

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
                qslnprob = qslogls * self.betas.reshape((self.ntemps, 1)) \
                    + qslogps

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

                accepts = accepts.reshape((self.ntemps, self.nwalkers/2))

                self.nprop[:, jupdate::2] += 1.0
                self.nprop_accepted[:, jupdate::2] += accepts

            p, lnprob, logl = self._temperature_swaps(p, lnprob, logl)

            if evolve_t and (i + 1) % self.evolution_time == 0:
                self._evolve_ladder(int(i / self.evolution_time))
                #pass

            if (i + 1) % thin == 0:
                if storechain:
                    self._chain[:, :, isave, :] = p
                    self._lnprob[:, :, isave, ] = lnprob
                    self._lnlikelihood[:, :, isave] = logl
                    isave += 1

            yield p, lnprob, logl

    def _evolve_ladder(self, t):
        descending = self.betas[-1] == 1
        if descending:
            # Temperatures are descending, so reverse them.
            self.betas = self.betas[::-1]

        lag = 500
        kappa = self.forcing_constant * lag / (t + lag)
        betas = self.betas.copy()

        As = self.tswap_acceptance_fraction_between_recent
        loggammas = -np.diff(np.log(self.betas))
        if self.target_acceptance != None:
            # Drive the chains to a specified acceptance ratio.

            # Compute new temperature spacings from acceptance spacings.
            As = np.concatenate(([self.target_acceptance], As))
            loggammas += kappa * (As[1:] - As[:-1])
        else:
            # Allow the chains to equilibrate to even acceptance-spacing for all chains.
            dlogbetas = np.zeros(len(self.betas))
            kappa = -kappa

            # Drive chains 1 to N-2 toward even sapcing
            dlogbetas[1:-2] = kappa * (As[:-1] - As[1:])
            #dlogbetas[-1] = kappa * (abs(As[-1] - As[-2]) - 0.1)

            # Require top two chains to achieve 100% acceptance with each other, but prevent them
            # from coalescing by driving upper chain faster.
            dlogbetas[-2:] = kappa * np.abs(1 - np.repeat(As[-1], 2)) * np.array([0.5, 1])

            # Compute new temperature spacings.
            loggammas += -np.diff(dlogbetas)

        # Ensure log-spacings are positive and adjust temperature chain. Whereever a negative spacing is
        # replaced by zero, must compensate by increasing subsequent spacing in order to preserve
        # higher temperatures.
        gaps = np.concatenate((-np.minimum(loggammas, 0)[:-1], [0]))
        loggammas = np.maximum(loggammas, 0) + np.roll(gaps, 1)
        self.betas[1:] = np.exp(-np.cumsum(loggammas))

        # Un-reverse the ladder if need be.
        if descending:
            self.betas = self.betas[::-1]

        # Store the ladder for reference.
        self._beta_history[:, t] = self.betas
        if callable(self.ladder_callback):
            self.ladder_callback(self)

        self.nswap_between_old = self.nswap_between.copy()
        self.nswap_between_old_accepted = self.nswap_between_accepted.copy()

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
    def tswap_acceptance_fraction_between_recent(self):
        """
        Returns an array of recently accepted temperature swap fractions for
        each pair of temperatures; shape ``(ntemps, )``.

        """
        return (self.nswap_between_accepted - self.nswap_between_old_accepted) /\
               (self.nswap_between - self.nswap_between_old)

