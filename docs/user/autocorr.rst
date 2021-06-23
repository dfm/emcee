.. _autocorr-user:

Autocorrelation Analysis
========================

A good heuristic for assessing convergence of samplings is the integrated
autocorrelation time. ``emcee`` includes tools for computing this and the
autocorrelation function itself. More details can be found in
:ref:`autocorr`.


.. autofunction:: emcee.autocorr.integrated_time
.. autofunction:: emcee.autocorr.function_1d
