.. module:: emcee

.. _sampler:

The Ensemble Sampler
====================

Standard usage of ``emcee`` involves instantiating an
:class:`EnsembleSampler`.

.. autoclass:: emcee.EnsembleSampler
   :inherited-members:

Note that several of the :class:`EnsembleSampler` methods return or consume
:class:`~emcee.state.State` objects:

.. autoclass:: emcee.state.State
   :inherited-members:

