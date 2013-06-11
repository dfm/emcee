#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["PTSampler"]

try:
    import acor
    acor = acor
except ImportError:
    acor = None

import emcee as em
import multiprocessing as multi
import numpy as np
import numpy.random as nr


class PTPost(object):
    """
    Wrapper for posterior used with the :class:`PTSampler` in emcee.

    """

    def __init__(self, logl, logp, beta):
        """
        :param logl:
            Function returning natural log of the likelihood.

        :param logp:
            Function returning natural log of the prior.

        :param beta:
            Inverse temperature of this chain: ``lnpost = beta*logl + logp``.

        """

        self._logl = logl
        self._logp = logp
        self._beta = beta

    def __call__(self, x):
        """
        Returns ``lnpost(x)``, ``lnlike(x)`` (the second value will be
        treated as a blob by emcee), where

        .. math::

            \ln \pi(x) \equiv \beta \ln l(x) + \ln p(x)

        :param x:
            The position in parameter space.

        """
        lp = self._logp(x)

        # If outside prior bounds, return 0.
        if lp == float('-inf'):
            return lp, lp

        ll = self._logl(x)

        return self._beta * ll + lp, ll


class PTSampler(em.Sampler):
    """
    A parallel-tempered ensemble sampler, using :class:`EnsembleSampler`
    for sampling within each parallel chain.

    :param ntemps:
        The number of temperatures.

    :param nwalkers:
        The number of ensemble walkers at each temperature.

    :param dim:
        The dimension of parameter space.

    :param logl:
        The log-likelihood function.

    :param logp:
        The log-prior function.

    :param threads: (optional)
        The number of parallel threads to use in sampling.

    :param pool: (optional)
        Alternative to ``threads``.  Any object that implements a
        ``map`` method compatible with the built-in ``map`` will do
        here.  For example, :class:`multi.Pool` will do.

    :param betas: (optional)
        Array giving the inverse temperatures, :math:`\\beta=1/T`, used in the
        ladder.  The default is for an exponential ladder, with beta
        decreasing by a factor of :math:`1/\\sqrt{2}` each rung.

    """
    def __init__(self, ntemps, nwalkers, dim, logl, logp, threads=1,
                 pool=None, betas=None):
        self.logl = logl
        self.logp = logp

        self.ntemps = ntemps
        self.nwalkers = nwalkers
        self.dim = dim

        self._chain = None
        self._lnprob = None
        self._lnlikelihood = None

        if betas is None:
            self._betas = self.exponential_beta_ladder(ntemps)
        else:
            self._betas = betas

        self.nswap = np.zeros(ntemps, dtype=np.float)
        self.nswap_accepted = np.zeros(ntemps, dtype=np.float)

        self.pool = pool
        if threads > 1 and pool is None:
            self.pool = multi.Pool(threads)

        self.samplers = [em.EnsembleSampler(nwalkers, dim,
                                            PTPost(logl, logp, b),
                                            pool=self.pool)
                                    for b in self.betas]

    def exponential_beta_ladder(self, ntemps):
        """
        Exponential ladder in :math:`1/T`, with :math:`T` increasing by
        :math:`\\sqrt{2}` each step, with ``ntemps`` in total.

        """
        return np.exp(np.linspace(0, -(ntemps - 1) * 0.5 * np.log(2), ntemps))

    def reset(self):
        """
        Clear the ``chain``, ``lnprobability``, ``lnlikelihood``,
        ``acceptance_fraction``, ``tswap_acceptance_fraction`` stored
        properties.

        """
        for s in self.samplers:
            s.reset()

        self.nswap = np.zeros(self.ntemps, dtype=np.float)
        self.nswap_accepted = np.zeros(self.ntemps, dtype=np.float)

        self._chain = None
        self._lnprob = None
        self._lnlikelihood = None

    def sample(self, p0, lnprob0=None, lnlike0=None, iterations=1,
               thin=1, storechain=True):
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
            lnprob0 = np.zeros((self.ntemps, self.nwalkers))
            lnlike0 = np.zeros((self.ntemps, self.nwalkers))
            for i in range(self.ntemps):
                fn = PTPost(self.logl, self.logp, self.betas[i])
                if self.pool is None:
                    results = list(map(fn, p[i, :, :]))
                else:
                    results = list(self.pool.map(fn, p[i, :, :]))

                lnprob0[i, :] = np.array([r[0] for r in results])
                lnlike0[i, :] = np.array([r[1] for r in results])

        lnprob = lnprob0
        logl = lnlike0

        # Expand the chain in advance of the iterations
        if storechain:
            nsave = iterations / thin
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

        for i in range(iterations):
            for j, s in enumerate(self.samplers):
                for psamp, lnprobsamp, rstatesamp, loglsamp in s.sample(
                                p[j, ...],
                                lnprob0=lnprob[j, ...],
                                blobs0=logl[j, ...], storechain=False):
                    p[j, ...] = psamp
                    lnprob[j, ...] = lnprobsamp
                    logl[j, ...] = np.array(loglsamp)

            p, lnprob, logl = self._temperature_swaps(p, lnprob, logl)

            if (i + 1) % thin == 0:
                if storechain:
                    self._chain[:, :, isave, :] = p
                    self._lnprob[:, :, isave, ] = lnprob
                    self._lnlikelihood[:, :, isave] = logl
                    isave += 1

            yield p, lnprob, logl

    def _temperature_swaps(self, p, lnprob, logl):
        """
        Perform parallel-tempering temperature swaps on the state
        in ``p`` with associated ``lnprob`` and ``logl``.

        """
        ntemps = self.ntemps

        for i in range(ntemps - 1, 0, -1):
            bi = self.betas[i]
            bi1 = self.betas[i - 1]

            dbeta = bi1 - bi

            iperm = nr.permutation(self.nwalkers)
            i1perm = nr.permutation(self.nwalkers)

            raccept = np.log(nr.uniform(size=self.nwalkers))
            paccept = dbeta * (logl[i, iperm] - logl[i - 1, i1perm])

            self.nswap[i] += self.nwalkers
            self.nswap[i - 1] += self.nwalkers

            asel = (paccept > raccept)
            nacc = np.sum(asel)

            self.nswap_accepted[i] += nacc
            self.nswap_accepted[i - 1] += nacc

            ptemp = np.copy(p[i, iperm[asel], :])
            ltemp = np.copy(logl[i, iperm[asel]])
            prtemp = np.copy(lnprob[i, iperm[asel]])

            p[i, iperm[asel], :] = p[i - 1, i1perm[asel], :]
            logl[i, iperm[asel]] = logl[i - 1, i1perm[asel]]
            lnprob[i, iperm[asel]] = lnprob[i - 1, i1perm[asel]] \
                                     - dbeta * logl[i - 1, i1perm[asel]]

            p[i - 1, i1perm[asel], :] = ptemp
            logl[i - 1, i1perm[asel]] = ltemp
            lnprob[i - 1, i1perm[asel]] = prtemp + dbeta * ltemp

        return p, lnprob, logl

    def thermodynamic_integration_log_evidence(self, logls=None, fburnin=0.1):
        """
        Thermodynamic integration estimate of the evidence.

        :param logls: (optional) The log-likelihoods to use for
            computing the thermodynamic evidence.  If ``None`` (the
            default), use the stored log-likelihoods in the sampler.
            Should be of shape ``(Ntemps, Nwalkers, Nsamples)``.

        :param fburnin: (optional)
            The fraction of the chain to discard as burnin samples; only the
            final ``1-fburnin`` fraction of the samples will be used to
            compute the evidence; the default is ``fburnin = 0.1``.

        :return ``(lnZ, dlnZ)``: Returns an estimate of the
            log-evidence and the error associated with the finite
            number of temperatures at which the posterior has been
            sampled.

        The evidence is the integral of the un-normalized posterior
        over all of parameter space:

        .. math::

            Z \\equiv \\int d\\theta \\, l(\\theta) p(\\theta)

        Thermodymanic integration is a technique for estimating the
        evidence integral using information from the chains at various
        temperatures.  Let

        .. math::

            Z(\\beta) = \\int d\\theta \\, l^\\beta(\\theta) p(\\theta)

        Then

        .. math::

            \\frac{d \\ln Z}{d \\beta}
            = \\frac{1}{Z(\\beta)} \\int d\\theta l^\\beta p \\ln l
            = \\left \\langle \\ln l \\right \\rangle_\\beta

        so

        .. math::

            \\ln Z(\\beta = 1)
            = \\int_0^1 d\\beta \\left \\langle \\ln l \\right\\rangle_\\beta

        By computing the average of the log-likelihood at the
        difference temperatures, the sampler can approximate the above
        integral.
        """

        if logls is None:
            return self.thermodynamic_integration_log_evidence(
                                    logls=self.lnlikelihood, fburnin=fburnin)
        else:
            betas = np.concatenate((self.betas, np.array([0])))
            betas2 = np.concatenate((self.betas[::2], np.array([0])))

            istart = int(logls.shape[2] * fburnin + 0.5)

            mean_logls = np.mean(np.mean(logls, axis=1)[:, istart:], axis=1)
            mean_logls2 = mean_logls[::2]

            lnZ = -np.dot(mean_logls, np.diff(betas))
            lnZ2 = -np.dot(mean_logls2, np.diff(betas2))

            return lnZ, np.abs(lnZ - lnZ2)

    @property
    def betas(self):
        """
        Returns the sequence of inverse temperatures in the ladder.

        """
        return self._betas

    @property
    def chain(self):
        """
        Returns the stored chain of samples; shape ``(Ntemps,
        Nwalkers, Nsteps, Ndim)``.

        """
        return self._chain

    @property
    def lnprobability(self):
        """
        Matrix of lnprobability values; shape ``(Ntemps, Nwalkers, Nsteps)``.

        """
        return self._lnprob

    @property
    def lnlikelihood(self):
        """
        Matrix of ln-likelihood values; shape ``(Ntemps, Nwalkers, Nsteps)``.

        """
        return self._lnlikelihood

    @property
    def tswap_acceptance_fraction(self):
        """
        Returns an array of accepted temperature swap fractions for
        each temperature; shape ``(ntemps, )``.

        """
        return self.nswap_accepted / self.nswap

    @property
    def acceptance_fraction(self):
        """
        Matrix of shape ``(Ntemps, Nwalkers)`` detailing the
        acceptance fraction for each walker.

        """
        return np.array([s.acceptance_fraction for s in self.samplers])

    @property
    def acor(self):
        """
        Returns a matrix of autocorrelation lengths for each
        parameter in each temperature of shape ``(Ntemps, Ndim)``.

        """
        if acor is None:
            raise ImportError('acor')
        else:
            acors = np.zeros((self.ntemps, self.dim))

            for i in range(self.ntemps):
                for j in range(self.dim):
                    acors[i, j] = acor.acor(self._chain[i, :, :, j])[0]

            return acors
