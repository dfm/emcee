.. _api:

API
===

.. module:: emcee

This page details the methods and classes provided by the ``emcee`` module.
The main entry point is through the :class:`EnsembleSampler` object.


The Affine-Invariant Ensemble Sampler
-------------------------------------

Standard usage of ``emcee`` involves instantiating an
:class:`EnsembleSampler` that contains a pair of :class:`Ensemble` objects
that trace out the Markov chain in parallel.

.. autoclass:: emcee.EnsembleSampler
   :inherited-members:

.. autoclass:: emcee.Ensemble
   :inherited-members:


Standard Metropolis-Hastings Sampler
------------------------------------

The `Metropolis-Hastings
<http://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>`_
sampler included in this module is far from fine-tuned and optimized. It
is, however, stable and it has a consistent API so it can be useful for
testing and comparison.

.. autoclass:: emcee.MHSampler
   :inherited-members:


Abstract Sampler Object
-----------------------

This section is mostly for developers who would be interested in implementing
a new sampler for inclusion in ``emcee``. A good starting point would be
to subclass the sampler object and override the :func:`Sampler.sample`
method.

.. autoclass:: emcee.Sampler
   :inherited-members:


Utilities
---------

.. autofunction:: emcee.utils.sample_ball
