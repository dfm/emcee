.. emcee documentation master file, created by
   sphinx-quickstart on Fri Jul 27 11:29:38 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

emcee
=====

Seriously Kick-Ass MCMC
-----------------------

.. raw:: html

    <p>Once upon a time, there was
    <a href="http://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm">an
    algorithm</a>. This algorithm became very popular but it turned out that
    there were better ones.
    <a href="http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml">Some folks</a>
    came up with a better one and <a href="http://danfm.ca">some</a>
    <a href="http://www.astro.princeton.edu/~dstn">other</a>
    <a href="http://cosmo.nyu.edu/hogg">folks</a> implemented it in Python.
    It's a little crazy how much it rocks.</p>


Basic Usage
-----------

If you wanted to draw samples from a 10 dimensional Gaussian, you would do
something like:

::

    import numpy as np
    import emcee

    def lnprob(x, ivar):
        return -0.5*np.sum(ivar * x**2)

    ndim, nwalkers = 10, 100
    ivar = 1./np.random.rand(ndim)
    p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[ivar])
    sampler.run_mcmc(p0, 1000)


User Guide
----------

.. toctree::
   :maxdepth: 2

   user/install
   user/quickstart


API Documentation
-----------------

.. toctree::
   :maxdepth: 2

   api


Contributors
------------

.. include:: ../AUTHORS.rst


Changelog
---------

.. include:: ../HISTORY.rst
