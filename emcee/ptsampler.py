try:
    import acor
    acor = acor
except ImportError:
    acor=None
import emcee as em
import multiprocessing as multi
import numpy as np
import numpy.random as nr

__all__ = ["PTSampler"]

class PTPost(object):
    """Wrapper for posterior used in emcee."""
    
    def __init__(self, logl, logp, beta):
        """Initialize with given log-likelihood, log-prior, and beta = 1/T."""

        self._logl = logl
        self._logp = logp
        self._beta = beta

    def __call__(self, x):
        """Returns lnpost(x), lnlike(x) (the second value will be
        treated as a blob by emcee), where lnpost(x) = beta*lnlike(x)
        + lnprior(x)."""

        lp = self._logp(x)

        # If outside prior bounds, return 0.
        if lp == float('-inf'):
            return lp, lp

        ll = self._logl(x)

        return self._beta*ll+lp, ll

class PTSampler(object):
    """A parallel-tempered ensemble sampler, using ``EnsembleSampler``
    for sampling within each parallel chain.

    :param ntemps: The number of temperatures.

    :param nwalkers: The number of ensemble walkers at each
        temperature.

    :param dim: The dimension of parameter space.

    :param logl: The ln(likelihood) function.

    :param logp: The ln(prior) function.

    :param threads: The number of parallel threads to use in sampling.

    :param pool: Alternative to ``threads``.  Any object that
        implements a ``map`` method compatible with the built-in
        ``map`` will do here.  For example, ``multiprocessing.Pool``.

    :param betas: Array giving the inverse temperature ladder,
    beta=1/T, used in the ladder.  The default is for an exponential
    ladder, with beta decreasing by a factor of 1/sqrt(2) each rung."""
    def __init__(self, ntemps, nwalkers, dim, logl, logp, threads=1, pool=None, betas=None):
        self.ntemps = ntemps
        self.nwalkers = nwalkers
        self.dim = dim

        if betas is None:
            self._betas = self.exponential_beta_ladder(ntemps)
        else:
            self._betas = betas

        self.nswap = np.zeros(ntemps, dtype=np.float)
        self.nswap_accepted = np.zeros(ntemps, dtype=np.float)

        self.pool = pool
        if threads > 1 and pool is None:
            self.pool = multi.Pool(threads)

        self.samplers = [em.EnsembleSampler(nwalkers, dim, PTPost(logl, logp, b), pool=self.pool) for b in self.betas]

    def exponential_beta_ladder(self, ntemps):
        """Exponential ladder in 1/T, nwith T increasing by sqrt(2)
        each step, with ``ntemps`` in total."""
        return np.exp(np.linspace(0, -(ntemps-1)*0.5*np.log(2), ntemps))

    def reset(self):
        """Clear the ``chain``, ``lnprobability``, ``lnlikelihood``,
        ``acceptance_fraction``, ``tswap_acceptance_fraction`` stored
        properties."""
    
        for s in self.samplers:
            s.reset()

    def sample(self, p0, lnprob0=None, logl0=None, iterations=1, storechain=True):
        """Advance the chains iterations steps as a generator.  
        
        :param p0: The initial positions of the walkers.  Shape should
        be ``(ntemps, nwalkers, dim)``.

        :param lnprob0: The initial ln(posterior) values for the
        ensembles.  Shape ``(ntemps, nwalkers)``.

        :param logl0: The initial ln(likelihood) values for the
        ensembles.  Shape ``(ntemps, nwalkers)``.

        :param iterations: The number of iterations to preform.

        :param storechain: If ``True`` store the iterations in the
        ``chain`` property.

        At each iteration, this generator yields

        * ``p``, the current position of the walkers.

        * ``lnprob`` the current ln(posterior) values for the walkers.

        * ``lnlike`` the current ln(likelihood) values for the walkers."""

        p = p0

        # If we have no lnprob or blobs, then run at least one
        # iteration to compute them.
        if lnprob0 is None or logl0 is None:
            iterations -= 1
            
            lnprob = []
            logl = []
            for i,s in enumerate(self.samplers):
                for psamp, lnprobsamp, rstatesamp, loglsamp in s.sample(p[i,...], storechain=storechain):
                    p[i,...] = psamp
                    lnprob.append(lnprobsamp)
                    logl.append(loglsamp)

            lnprob = np.array(lnprob) # Dimensions (ntemps, nwalkers)
            logl = np.array(logl)

            p,lnprob,logl = self.temperature_swaps(p, lnprob, logl)
        else:
            lnprob = lnprob0
            logl = logl0

        for i in range(iterations):
            for i,s in enumerate(self.samplers):
                for psamp, lnprobsamp, rstatesamp, loglsamp in s.sample(p[i,...], lnprob0=lnprob[i,...], blobs0=logl[i,...], storechain=storechain):
                    p[i,...] = psamp
                    lnprob[i,...] = lnprobsamp
                    logl[i,...] = np.array(loglsamp)

            p,lnprob,logl = self.temperature_swaps(p, lnprob, logl)

            yield p, lnprob, logl

    def temperature_swaps(self, p, lnprob, logl):
        """Perform parallel-tempering temperature swaps on the state
        in p with associated lnprob and logl."""

        ntemps=self.ntemps

        for i in range(ntemps-1, 0, -1):
            bi=self.betas[i]
            bi1=self.betas[i-1]

            dbeta = bi1-bi

            for j in range(self.nwalkers):
                self.nswap[i] += 1
                self.nswap[i-1] += 1

                ii=nr.randint(self.nwalkers)
                jj=nr.randint(self.nwalkers)

                paccept = dbeta*(logl[i, ii] - logl[i-1, jj])

                if paccept > 0 or np.log(nr.rand()) < paccept:
                    self.nswap_accepted[i] += 1
                    self.nswap_accepted[i-1] += 1

                    ptemp=np.copy(p[i, ii, :])
                    logltemp=logl[i, ii]
                    lnprobtemp=lnprob[i, ii]

                    p[i,ii,:]=p[i-1,jj,:]
                    logl[i,ii]=logl[i-1, jj]
                    lnprob[i,ii] = lnprob[i-1,jj] - dbeta*logl[i-1,jj]

                    p[i-1,jj,:]=ptemp
                    logl[i-1,jj]=logltemp
                    lnprob[i-1,jj]=lnprobtemp + dbeta*logltemp

        return p, lnprob, logl

    @property
    def betas(self):
        """Returns the sequence of inverse temperatures in the
        ladder."""
        return self._betas

    @property 
    def chain(self):
        """Returns the stored chain of samples.  Will have shape that
        is ``(Ntemps, Nwalkers, Nsteps, Ndim)``."""

        return np.array([s.chain for s in self.samplers])

    @property
    def lnprobability(self):
        """Matrix of lnprobability values, of shape ``(Ntemps, Nwalkers, Nsteps)``"""
        return np.array([s.lnprobability for s in self.samplers])

    @property
    def lnlikelihood(self):
        """Matrix of ln-likelihood values of shape ``(Ntemps, Nwalkers, Nsteps)``."""
        return np.array([np.transpose(np.array(s.blobs)) for s in self.samplers])

    @property
    def tswap_acceptance_fraction(self):
        """Returns an array of accepted temperature swap fractions for
        each temperature, shape ``(ntemps, )``."""
        return self.nswap_accepted / self.nswap

    @property
    def acceptance_fraction(self):
        """Matrix of shape ``(Ntemps, Nwalkers)`` detailing the acceptance
        fraction for each walker."""
        return np.array([s.acceptance_fraction for s in self.samplers])

    @property
    def acor(self):
        """Returns a matrix of autocorrelation lengths for each
        parameter in each temperature of shape ``(Ntemps, Ndim)``."""
        return np.array([s.acor for s in self.samplers])

