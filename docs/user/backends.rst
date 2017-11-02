.. module:: emcee

.. _backends:

Backends
========

Starting with version 0.3, emcee has an interface for serializing the sampler
output.
This can be useful in any scenario where you want to share the results of
sampling or when sampling with an expensive model because, even if the
sampler crashes, the current state of the chain will always be saved.

.. autoclass:: emcee.backends.Backend
   :inherited-members:

.. autoclass:: emcee.backends.HDFBackend
   :inherited-members:
