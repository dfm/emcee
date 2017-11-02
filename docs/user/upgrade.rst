.. module:: emcee

.. _upgrade:

Upgrading From Pre-3.0 Versions
===============================

The version 3 release of emcee is the biggest update in years.
That being said, we've made every attempt to maintain backwards compatibility
while still offering new features.
The main new features include:

1. A :ref:`moves` interface that allows the use of a variety of ensemble
   proposals,

2. A more self consistent and user-friendly :ref:`blobs` interface,

3. A :ref:`backends` interface that simplifies the process of serializing the
   sampling results, and

4. The long requested progress bar (implemented using `tqdm
   <https://github.com/tqdm/tqdm>`_) so that users can watch the grass grow
   while the sampler does its thing (this is as simple as installing tqdm and
   setting ``progress=True`` in :class:`EnsembleSampler`).
