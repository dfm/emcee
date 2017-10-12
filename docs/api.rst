.. _api:

API
===

.. module:: emcee

This page details the methods and classes provided by the ``emcee`` module.
The main entry point is through the :class:`EnsembleSampler` object.


The Ensemble Sampler
--------------------

Standard usage of ``emcee`` involves instantiating an
:class:`EnsembleSampler`.

.. autoclass:: emcee.EnsembleSampler
   :inherited-members:


Moves
-----

.. autoclass:: emcee.moves.StretchMove
    :inherited-members:


Autocorrelation Analysis
------------------------

A good heuristic for assessing convergence of samplings is the integrated
autocorrelation time. ``emcee`` includes tools for computing this and the
autocorrelation function itself. More details can be found in
:ref:`autocorr`.


.. autofunction:: emcee.autocorr.integrated_time
.. autofunction:: emcee.autocorr.function_1d


Interruptible Pool
------------------

.. automodule:: emcee.interruptible_pool
.. autoclass:: emcee.interruptible_pool.InterruptiblePool
   :members:
