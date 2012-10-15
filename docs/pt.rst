.. _pt:

.. module:: emcee


Parallel-Tempering Ensemble MCMC
================================

When your posterior is multi-modal or otherwise hard to sample with a
standard MCMC, try a parallel-tempered MCMC (PTMCMC).  A PTMCMC runs
multiple MCMC's at different temperatures, :math:`T`.  Each MCMC
samples from a modified posterior, given by

.. math::

   \pi_T(x) = \left[ l(x) \right]^{\frac{1}{T}} p(x)

As :math:`T \to \infty`, the posterior becomes the prior, which is
hopefully easy to sample.  If the likelihood is a Gaussian with
standard deviation :math:`\sigma`, then the tempered likelihood is
proportional to a Gaussian with standard deviation :math:`\sigma
\sqrt{T}`.

Periodically during the run, the different temperatures swap members
of their ensemble in a way that preserves detailed balance.  The hot
chains can more easily explore parameter space because the likelihood
is flatter and broader, while the cold chains do a good job of
exploring the peaks of the likelihood.  This can **dramatically**
improve convergence if your likelihood function has many
well-separated modes.

How To Sample a Multi-Modal Gaussian
------------------------------------

Suppose we want to sample from the posterior given by 

.. math::

   \pi(\vec{x}) \propto \exp\left[ - \frac{1}{2} \left( \vec{x} - \vec{mu}_1 \right)^T \Sigma^{-1}_1 \left( \vec{x} - \vec{\mu}_1 \right) \right] + \exp\left[ -\frac{1}{2} \left( \vec{x} - \vec{\mu}_2 \right)^T \Sigma^{-1}_2 \left( \vec{x} - \vec{\mu}_2 \right) \right]

If the modes :math:`\mu_{1,2}` are well-separated with respect to the
scale of :math:`\Sigma_{1,2}`, then this distribution will be hard to
sample with the :class:`EnsembleSampler`.  Here is how we would sample
from it using the :class:`PTSampler`.

First, some preliminaries:

::

    import numpy as np
    from emcee import PTSampler

Define the means and standard deviations of our multi-modal likelihood:

::

    # mu1 = [1, 1], mu2 = [-1, -1]
    mu1 = np.ones(2)
    mu2 = -np.ones(2)

    # Width of 0.1 in each dimension
    sigma1inv = np.diag([100.0, 100.0])
    sigma2inv = np.diag([100.0, 100.0])

    def logl(x):
        dx1 = x - mu1
	dx2 = x - mu2

	return np.logaddexp(-np.dot(dx1, np.dot(sigma1inv, dx1))/2.0,
	                    -np.dot(dx2, np.dot(sigma2inv, dx2))/2.0)

    # Use a flat prior
    def logp(x):
        return 0.0

Now we can construct a sampler object that will drive the PTMCMC;
arbitrarily, we choose to use 20 temperatures (the default is for each
temperature to increase by a factor of :math:`\sqrt{2}`, so the
highest temperature will be :math:`T = 1024`, resulting in an
effective :math:`\sigma_T = 32 \sigma = 3.2`, which is about the
separation of our modes).  Let's use 100 walkers in the ensemble at
each temperature, and run with 2 parallel threads.

::

    ntemps = 20
    nwalkers = 100
    ndim = 2
    nthreads = 2

    sampler=PTSampler(ntemps, nwalkers, ndim, logl, logp, threads=nthreads)

We could have modified the temperature ladder using the ``betas``
optional argument (which should be an array of :math:`\beta \equiv 1/T` values).
The ``pool`` argument also allows to specify our own pool
of worker threads if we wanted fine-grained control over the
parallelism.

First, we run the sampler for 1000 burn-in iterations::

    for p, lnprob, lnlike in sampler.sample(np.random.uniform(low=-1.0, high=1.0, size=(ntemps, nwalkers, ndim))):
        pass
    sampler.reset()

Now we sample for 10000 iterations, recording every 10th sample::

    for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob, logl0=lnlike, iterations=10000, thin=10):
        pass

The resulting samples (1000 of them) are stored as the
``sampler.chain`` property::

    # Chain has shape (ntemps, nwalkers, nsteps, ndim)
    # Zero temperature mean:
    mu0 = np.mean(np.mean(sampler.chain[0,...], axis=0), axis=0)

    # Longest autocorrelation length (over any temperature)
    max_acl = np.max(sampler.acor)
