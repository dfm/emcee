.. _api:

API
===

.. module:: emcee

This page details the methods and classes provided by the ``emcee`` module.
The main entry point is through the :class:`EnsembleSampler` object.


The Affine-Invariant Ensemble Sampler
-------------------------------------

Standard usage of ``emcee`` involves instantiating an
:class:`EnsembleSampler`.

.. autoclass:: emcee.EnsembleSampler
   :inherited-members:

The Parallel-Tempered Ensemble Sampler
--------------------------------------

The :class:`PTSampler` class performs a parallel-tempered ensemble
sampling using :class:`EnsembleSampler` to sample within each
temperature.  This sort of sampling is useful if you expect your
distribution to be multi-modal. Take a look at :doc:`the documentation
</user/pt>` to see how you might use this class.

.. autoclass:: emcee.PTSampler
   :members:
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


Autocorrelation Analysis
------------------------

A good heuristic for assessing convergence of samplings is the integrated
autocorrelation time. ``emcee`` includes (as of version 2.1.0) tools for
computing this and the autocorrelation function itself.

.. autofunction:: emcee.autocorr.integrated_time

.. autofunction:: emcee.autocorr.function


Utilities
---------

.. autofunction:: emcee.utils.sample_ball

.. autoclass:: emcee.utils.MH_proposal_axisaligned

Pools
-----

These are some helper classes for using the built-in parallel version of the
algorithm. These objects can be initialized and then passed into the
constructor for the :class:`EnsembleSampler` object using the ``pool`` keyword
argument.

Interruptible Pool
++++++++++++++++++

.. automodule:: emcee.interruptible_pool

.. autoclass:: emcee.interruptible_pool.InterruptiblePool
   :members:

MPI Pool
++++++++

Built-in support for MPI distributed systems. See the documentation:
:ref:`mpi`.

.. autoclass:: emcee.utils.MPIPool
   :members:
