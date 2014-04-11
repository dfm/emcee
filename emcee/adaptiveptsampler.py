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
    def __init__(self, *args, ladder_callback=None, target_acceptance=0.25, evolution_time=100, forcing_constant=20, **kwargs):
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

        # If we have no lnprob or logls compute them
        if lnprob0 is None or lnlike0 is None:
            fn = PTLikePrior(self.logl, self.logp, self.loglargs,
                             self.logpargs, self.loglkwargs, self.logpkwargs)
            if self.pool is None:
                results = list(map(fn, p.reshape((-1, self.dim))))
            else:
                results = list(self.pool.map(fn, p.reshape((-1, self.dim))))

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
                if self.pool is None:
                    results = list(map(fn, qs.reshape((-1, self.dim))))
                else:
                    results = list(self.pool.map(fn, qs.reshape((-1,
                                                                 self.dim))))

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

            if (i + 1) % thin == 0:
                if storechain:
                    self._chain[:, :, isave, :] = p
                    self._lnprob[:, :, isave, ] = lnprob
                    self._lnlikelihood[:, :, isave] = logl
                    isave += 1

            yield p, lnprob, logl

    def _evolve_ladder(self, step):
        descending = self.betas[-1] == 1
        if descending:
            # Temperatures are desending, so reverse them.
            self.betas = self.betas[::-1]

        # Compute forcing terms for each gamma.
        A0 = self.target_acceptance
        lag = 120
        kappa = self.forcing_constant * lag / (step + lag)

        # Get swap acceptance fractions (prepending target fraction) and temperature ratios.
        As = np.concatenate(([A0], self.tswap_acceptance_fraction_between_recent))
        gammas = np.exp(-np.diff(np.log(self.betas)))

        # Adjust log(gamma) by difference between corresponding acceptance and next lowest
        # acceptance. Cut off below 1 to prevent weird temperature behavior.
        dloggammas = kappa * (As[1:] - As[:-1])
        gammas *= np.exp(dloggammas)
        gammas = np.maximum(gammas, 1)

        # Work upwards from the bottom to adjust temperature ladder.
        for i in range(len(self.betas) - 1):
            self.betas[i + 1] = self.betas[i] / gammas[i]

        # Un-reverse the ladder if need be.
        if descending:
            self.betas = self.betas[::-1]

        # Store the ladder for reference.
        self._beta_history[:, step] = self.betas
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

    def _add_temperature(self, beta):
        # First, reset the sampler. 
        self.reset()

        # Add the new beta wherever it needs to be.
        indices = np.where(self.betas > beta)[0]
        if len(index) > 0:
            np.insert(self.betas, indices[0], beta)
        else:
            np.append(self.betas, beta)

        # Place the new chain in a point sampled from the prior.
        raise Exception

        return 

    @property
    def tswap_acceptance_fraction_between_recent(self):
        """
        Returns an array of recently accepted temperature swap fractions for
        each pair of temperatures; shape ``(ntemps, )``.

        """
        return (self.nswap_between_accepted - self.nswap_between_old_accepted) /\
               (self.nswap_between - self.nswap_between_old)

