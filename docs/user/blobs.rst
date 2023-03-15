.. _blobs:

Blobs
=====

Way back in version 1.1 of emcee, the concept of blobs was introduced.
This allows a user to track arbitrary metadata associated with every sample in
the chain.
The interface to access these blobs was previously a little clunky because it
was stored as a list of lists of blobs.
In version 3, this interface has been updated to use NumPy arrays instead and
the sampler will do type inference to save the simplest possible
representation of the blobs.

Anything that your ``log_prob`` function returns in addition to the log
probability is assumed to be a blob and is tracked as part of blobs.
Put another way, if ``log_prob`` returns more than one thing, all the things
after the first (which is assumed to be the log probability) are assumed to be
blobs.
If ``log_prob`` returns ``-np.inf`` for the log probability, the blobs are not
inspected or tracked so can be anything (but the correct number of arguments
must still be returned).

Using blobs to track the value of the prior
-------------------------------------------

A common pattern is to save the value of the log prior at every step in the
chain.
To do this, your ``log_prob`` function should return your blobs (in this case log prior) as well as the log probability when called.
This can be implemented something like:

.. code-block:: python

    import emcee
    import numpy as np

    def log_prior(params):
        return -0.5 * np.sum(params**2)

    def log_like(params):
        return -0.5 * np.sum((params / 0.1)**2)

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp):
            # log prior is not finite, return -np.inf for log probability
            # and None for log prior as it won't be used anyway (but we
            # must use the correct number of return values)
            return -np.inf, None
        ll = log_like(params)
        if not np.isfinite(ll):
            # log likelihood is not finite, return -np.inf for log
            # probability and None for log prior (again, this value isn't
            # used but we have to have the correct number of return values)
            return -np.inf, None

        # return log probability (sum of log prior and log likelihood)
        # and log prior. Log prior will be saved as part of the blobs.
        return lp + ll, lp

    coords = np.random.randn(32, 3)
    nwalkers, ndim = coords.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(coords, 100)

    log_prior_samps = sampler.get_blobs()
    flat_log_prior_samps = sampler.get_blobs(flat=True)

    print(log_prior_samps.shape)  # (100, 32)
    print(flat_log_prior_samps.shape)  # (3200,)

As shown above, after running this, the "blobs" stored by the sampler will be
a ``(nsteps, nwalkers)`` NumPy array with the value of the log prior at every
sample.

Named blobs & custom dtypes
---------------------------

If you want to save multiple pieces of metadata, it can be useful to name
them.
To implement this, we use the ``blobs_dtype`` argument in
:class:`EnsembleSampler`.
Using this is also helpful to specify types.
If you don't provide ``blobs_dtype``, the dtype of the extra args is automatically guessed the first time ``log_prob`` is called.

For example, let's say that, for some reason, we wanted to save the mean of
the parameters as well as the log prior.
To do this, we would update the above example as follows:

.. code-block:: python

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp):
            # As above, log prior is not finite, so return -np.inf for log
            # probability and None for everything else (these values aren't
            # used, but the number of return values must be correct)
            return -np.inf, None, None
        ll = log_like(params)
        if not np.isfinite(ll):
            # Log likelihood is not finite so return -np.inf for log
            # probabilitiy and None for everything else (maintaining the
            # correct number of return values)
            return -np.inf, None, None

        # All values are finite, so return desired blobs (in this case: log
        # probability, log prior and mean of parameters)
        return lp + ll, lp, np.mean(params)

    coords = np.random.randn(32, 3)
    nwalkers, ndim = coords.shape

    # Here are the important lines for defining the blobs_dtype
    dtype = [("log_prior", float), ("mean", float)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                    blobs_dtype=dtype)

    sampler.run_mcmc(coords, 100)

    blobs = sampler.get_blobs()
    log_prior_samps = blobs["log_prior"]
    mean_samps = blobs["mean"]
    print(log_prior_samps.shape)  # (100, 32)
    print(mean_samps.shape)  # (100, 32)

    flat_blobs = sampler.get_blobs(flat=True)
    flat_log_prior_samps = flat_blobs["log_prior"]
    flat_mean_samps = flat_blobs["mean"]
    print(flat_log_prior_samps.shape)  # (3200,)
    print(flat_mean_samps.shape)  # (3200,)

This will print

.. code-block:: python

    (100, 32)
    (100, 32)
    (3200,)
    (3200,)

and the ``blobs`` object will be a structured NumPy array with two columns
called ``log_prior`` and ``mean``.
