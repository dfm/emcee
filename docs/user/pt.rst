.. _pt:

.. module:: emcee


Parallel-Tempering Ensemble MCMC
================================

*Added in version 1.2.0*

When your posterior is multi-modal or otherwise hard to sample with a
standard MCMC, a good option to try is `parallel-tempered MCMC (PTMCMC)
<http://en.wikipedia.org/wiki/Parallel_tempering>`_.
PTMCMC runs multiple MCMC's at different temperatures, :math:`T`.  Each MCMC
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

   \pi(\vec{x}) \propto \exp\left[ - \frac{1}{2}
        \left( \vec{x} - \vec{\mu}_1 \right)^T \Sigma^{-1}_1
        \left( \vec{x} - \vec{\mu}_1 \right) \right]
        + \exp\left[ -\frac{1}{2} \left( \vec{x} - \vec{\mu}_2 \right)^T
          \Sigma^{-1}_2 \left( \vec{x} - \vec{\mu}_2 \right) \right]

If the modes :math:`\mu_{1,2}` are well-separated with respect to the
scale of :math:`\Sigma_{1,2}`, then this distribution will be hard to
sample with the :class:`EnsembleSampler`.  Here is how we would sample
from it using the :class:`PTSampler`.

First, some preliminaries::

    import numpy as np
    from emcee import PTSampler

Define the means and standard deviations of our multi-modal likelihood::

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
each temperature::

    ntemps = 20
    nwalkers = 100
    ndim = 2

    sampler=PTSampler(ntemps, nwalkers, ndim, logl, logp)

Making the sampling multi-threaded is as simple as adding the
``threads=Nthreads`` argument to :class:`PTSampler`.  We could have
modified the temperature ladder using the ``betas`` optional argument
(which should be an array of :math:`\beta \equiv 1/T` values).  The
``pool`` argument also allows to specify our own pool of worker
threads if we wanted fine-grained control over the parallelism.

First, we run the sampler for 1000 burn-in iterations::

    p0 = np.random.uniform(low=-1.0, high=1.0, size=(ntemps, nwalkers, ndim))
    for p, lnprob, lnlike in sampler.sample(p0, iterations=1000):
        pass
    sampler.reset()

Now we sample for 10000 iterations, recording every 10th sample::

    for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
                                               lnlike0=lnlike,
                                               iterations=10000, thin=10):
        pass

The resulting samples (1000 of them) are stored as the
``sampler.chain`` property::

    assert sampler.chain.shape == (ntemps, nwalkers, 1000, ndim)

    # Chain has shape (ntemps, nwalkers, nsteps, ndim)
    # Zero temperature mean:
    mu0 = np.mean(np.mean(sampler.chain[0,...], axis=0), axis=0)

    # Longest autocorrelation length (over any temperature)
    max_acl = np.max(sampler.acor)

    # etc


Implementation Notes
--------------------

For a description of the parallel-tempering algorithm, see, e.g. `Earl
& Deem (2010), Phys Chem Chem Phys, 7, 23, 3910
<http://adsabs.harvard.edu/abs/2005PCCP....7.3910E>`_. The algorithm
has some tunable parameters:

Temperature Ladder
    The choice of temperature for the chains will strongly influence
    the rate of convergence of the sampling.  By default, the
    :class:`PTSampler` class uses an exponential ladder, with each
    temperature increasing by a factor of :math:`\sqrt{2}`.  The user
    can supply their own ladder using the ``beta`` optional argument
    in the constructor.
Rate of Temperature Swaps
    The rate at which temperature swaps are proposed can, to a lesser
    extent, also influence the rate of convergence of the sampling.
    The goal is to make sure that good positions found by the high
    temperatures can propogate to the lower temperatures, but still
    ensure that the high-temperatures do not lose all memory of good
    locations.  Here we choose to implement one temperature swap
    proposal per walker per rung on the temperature ladder after each
    ensemble update.  This is not user-tunable, but seems to work well
    in practice.

The ``args`` optional argument is not available in the
:class:`PTSampler` constructor; use a custom class with a ``__call__``
method if you need to pass arguments to the ``lnlike`` or ``lnprior``
functions and do not want to use a global variable.

The ``thermodynamic_integration_log_evidence`` method uses
thermodynamic integration (see, e.g., `Goggans & Chi (2004), AIP Conf
Proc, 707, 59 <http://dx.doi.org/10.1063/1.1751356>`_) to estimate the
evidence integral.  Note that thermodynamic integration requires a
proper prior.  For example, even though the multimodal distribution
above has a well-defined evidence integral, it cannot be computed
through thermodynamic integration because the prior is improper.  A
simple change in the prior to cut it off at :math:`\pm 10\sigma` would
enable thermodynamic integration to proceed without meaningfully
changing the evidence integral.

Define the evidence as a function of inverse temperature:

.. math::

    Z(\beta) \equiv \int dx\, l^\beta(x) p(x)

We want to compute :math:`Z(1)`.  :math:`Z` satisfies the following
differential equation

.. math::

    \frac{d \ln Z}{d\beta}
        = \frac{1}{Z(\beta)} \int dx\, \ln l(x) l^\beta(x) p(x)
        = \left \langle \ln l \right\rangle_\beta

where :math:`\left\langle \ldots \right\rangle_\beta` is the average
of a quantity over the posterior at temperature :math:`T = 1/\beta`.
Integrating (note that :math:`Z(0) = 1` because the prior is
normalized), we have

.. math::

    \ln Z(1) = \int_0^1 d\beta \left \langle \ln l \right\rangle_\beta

This quantity can be estimated from a PTMCMC by computing the average
:math:`\ln l` within each chain and applying a quadrature formula to
estimate the integral.
