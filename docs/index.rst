emcee
=====

Seriously Kick-Ass MCMC
-----------------------

``emcee`` is an MIT licensed pure-Python implementation of Goodman & Weare's
`Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler
<http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml>`_ and these pages will
show you how to use it.

This documentation won't teach you too much about MCMC but there are a lot
of resources available for that (try `this one
<http://www.inference.phy.cam.ac.uk/mackay/itila/book.html>`_).
We also `published a paper <http://arxiv.org/abs/1202.3665>`_ explaining
the ``emcee`` algorithm and implementation in detail.

emcee has been used in `quite a few projects in the astrophysical literature
<testimonials>`_ and it is being actively developed on `GitHub
<https://github.com/dfm/emcee>`_.


Basic Usage
-----------

If you wanted to draw samples from a 10 dimensional Gaussian, you would do
something like:

.. code-block:: python

    import numpy as np
    import emcee

    def lnprob(x, ivar):
        return -0.5 * np.sum(ivar * x ** 2)

    ndim, nwalkers = 10, 100
    ivar = 1. / np.random.rand(ndim)
    p0 = [np.random.rand(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[ivar])
    sampler.run_mcmc(p0, 1000)

A more complete example is available in the `quickstart documentation
<user/quickstart>`_.


User Guide
----------

.. toctree::
   :maxdepth: 2

   user/install
   user/quickstart
   user/line
   user/advanced
   user/pt
   user/faq


API Documentation
-----------------

.. toctree::
   :maxdepth: 2

   api


Contributors
------------

.. include:: ../AUTHORS.rst


License & Attribution
---------------------

Copyright 2010-2016 Dan Foreman-Mackey and contributors.

emcee is free software made available under the MIT License. For details
see `LICENSE <license>`_.

If you make use of emcee in your work, please cite our paper
(`arXiv <http://arxiv.org/abs/1202.3665>`_,
`ADS <http://adsabs.harvard.edu/abs/2013PASP..125..306F>`_,
`BibTeX <http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2013PASP..125..306F&data_type=BIBTEX>`_)
and consider adding your paper to the :ref:`testimonials` list.


Changelog
---------

.. include:: ../HISTORY.rst
