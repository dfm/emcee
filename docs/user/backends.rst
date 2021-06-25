.. _backends:

Backends
========

Starting with version 3, emcee has an interface for serializing the sampler
output.
This can be useful in any scenario where you want to share the results of
sampling or when sampling with an expensive model because, even if the
sampler crashes, the current state of the chain will always be saved.

There is currently one backend that can be used to serialize the chain to a
file: :class:`emcee.backends.HDFBackend`.
The methods and options for this backend are documented below.
It can also be used as a reader for existing samplings.
For example, if a chain was saved using the :class:`backends.HDFBackend`, the
results can be accessed as follows:

.. code-block:: python

    reader = emcee.backends.HDFBackend("chain_filename.h5", read_only=True)
    flatchain = reader.get_chain(flat=True)

The ``read_only`` argument is not required, but it will make sure that you
don't inadvertently overwrite the samples in the file.

.. autoclass:: emcee.backends.Backend
   :inherited-members:

.. autoclass:: emcee.backends.HDFBackend
   :inherited-members:
