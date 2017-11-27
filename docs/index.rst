emcee
=====

**emcee** is an MIT licensed pure-Python implementation of Goodman & Weare's
`Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler
<http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml>`_ and these pages will
show you how to use it.

This documentation won't teach you too much about MCMC but there are a lot
of resources available for that (try `this one
<http://www.inference.org.uk/mackay/itprnn/book.html>`_).
We also `published a paper <http://arxiv.org/abs/1202.3665>`_ explaining
the emcee algorithm and implementation in detail.

emcee has been used in `quite a few projects in the astrophysical literature
<testimonials>`_ and it is being actively developed on `GitHub
<https://github.com/dfm/emcee>`_.

.. image:: https://img.shields.io/badge/GitHub-dfm%2Femcee-blue.svg?style=flat
    :target: https://github.com/dfm/emcee
.. image:: http://img.shields.io/travis/dfm/emcee/master.svg?style=flat
    :target: http://travis-ci.org/dfm/emcee
.. image:: https://ci.appveyor.com/api/projects/status/p8smxvleh8mrcn6m?svg=true&style=flat
    :target: https://ci.appveyor.com/project/dfm/emcee
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/dfm/emcee/blob/master/LICENSE
.. image:: http://img.shields.io/badge/arXiv-1202.3665-orange.svg?style=flat
    :target: http://arxiv.org/abs/1202.3665
.. image:: https://coveralls.io/repos/github/dfm/emcee/badge.svg?branch=master&style=flat
    :target: https://coveralls.io/github/dfm/emcee?branch=master


Basic Usage
-----------

If you wanted to draw samples from a 5 dimensional Gaussian, you would do
something like:

.. code-block:: python

    import numpy as np
    import emcee

    def log_prob(x, ivar):
        return -0.5 * np.sum(ivar * x ** 2)

    ndim, nwalkers = 5, 100
    ivar = 1. / np.random.rand(ndim)
    p0 = np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[ivar])
    sampler.run_mcmc(p0, 10000)

A more complete example is available in the :ref:`quickstart` tutorial.


How to Use This Guide
---------------------

To start, you're probably going to need to follow the :ref:`install` guide to
get emcee installed on your computer.
After you finish that, you can probably learn most of what you need from the
from the tutorials listed below (you might want to start with
:ref:`quickstart` and go form there).
If you need more details about specific functionality, the User Guide below
should have what you need.
If you run into any issues, please don't hesitate to `open an issue on GitHub
<https://github.com/dfm/emcee/issues>`_.


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/install
   user/sampler
   user/moves
   user/blobs
   user/backends
   user/autocorr
   user/upgrade
   user/faq

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/quickstart
   tutorials/line
   tutorials/parallel
   tutorials/autocorr
   tutorials/monitor


Contributors
------------

.. include:: ../AUTHORS.rst


License & Attribution
---------------------

Copyright 2010-2017 Dan Foreman-Mackey and contributors.

emcee is free software made available under the MIT License. For details
see the ``LICENSE``.

If you make use of emcee in your work, please cite our paper
(`arXiv <http://arxiv.org/abs/1202.3665>`_,
`ADS <http://adsabs.harvard.edu/abs/2013PASP..125..306F>`_,
`BibTeX <http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2013PASP..125..306F&data_type=BIBTEX>`_)
and consider adding your paper to the :ref:`testimonials` list.


Changelog
---------

.. include:: ../HISTORY.rst
