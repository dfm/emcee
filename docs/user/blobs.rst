.. module:: emcee

.. _blobs:

Blobs
=====

Way back in version 1.1 of emcee, the concept of blobs was introduced.
This allows a user to track arbitrary metadata associated with every sample in
the chain.
The interface to access these blobs was previously a little clunky because it
was stored as a list of lists of blobs.
In version 3, this interface has been updated to use NumPy arrays instead and
the sampler will do type inference to save the simplest possible
representation of the blobs.
