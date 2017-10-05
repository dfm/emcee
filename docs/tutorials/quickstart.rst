
.. module:: george

**Note:** This tutorial was generated from an IPython notebook that can be
downloaded `here <../../_static/notebooks/quickstart.ipynb>`_.

.. _quickstart:


Quickstart
==========

This notebook was made with the following version of emcee:

.. code:: python

    import emcee
    emcee.__version__




.. parsed-literal::

    '3.0.0.dev0'



The easiest way to get started with using emcee is to use it for a
project. To get you started, here’s an annotated, fully-functional
example that demonstrates a standard usage pattern.

How to sample a multi-dimensional Gaussian
------------------------------------------

We’re going to demonstrate how you might draw samples from the
multivariate Gaussian density given by:

.. math::


   p(\vec{x}) \propto \exp \left [ - \frac{1}{2} (\vec{x} -
       \vec{\mu})^\mathrm{T} \, \Sigma ^{-1} \, (\vec{x} - \vec{\mu})
       \right ]

where :math:`\vec{\mu}` is an :math:`N`-dimensional vector position of
the mean of the density and :math:`\Sigma` is the square N-by-N
covariance matrix.

The first thing that we need to do is import the necessary modules:

.. code:: python

    import numpy as np
    import emcee

Then, we’ll code up a Python function that returns the density
:math:`p(\vec{x})` for specific values of :math:`\vec{x}`,
:math:`\vec{\mu}` and :math:`\Sigma^{-1}`. In fact, emcee actually
requires the logarithm of :math:`p`. We’ll call it ``log_prob``:

.. code:: python

    def log_prob(x, mu, icov):
        diff = x - mu
        return -0.5*np.dot(diff,np.dot(icov,diff))

