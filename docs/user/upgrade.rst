.. _upgrade:

Upgrading From Pre-3.0 Versions
===============================

The version 3 release of emcee is the biggest update in years.
That being said, we've made every attempt to maintain backwards compatibility
while still offering new features.
The main new features include:

1. A :ref:`moves-user` interface that allows the use of a variety of ensemble
   proposals,

2. A more self consistent and user-friendly :ref:`blobs` interface,

3. A :ref:`backends` interface that simplifies the process of serializing the
   sampling results, and

4. The long requested progress bar (implemented using `tqdm
   <https://github.com/tqdm/tqdm>`_) so that users can watch the grass grow
   while the sampler does its thing (this is as simple as installing tqdm and
   setting ``progress=True`` in :class:`EnsembleSampler`).

To improve the stability and supportability of emcee, we also removed some
features.
The main removals are as follows:

1. The ``threads`` keyword argument has been removed in favor of the ``pool``
   interface (see the :ref:`parallel` tutorial for more information).
   The old interface had issues with memory consumption and hanging processes
   when the sampler object wasn't explicitly deleted.
   The ``pool`` interface has been supported since the first release of emcee
   and existing code should be easy to update following the :ref:`parallel`
   tutorial.

2. The ``MPIPool`` has been removed and forked to the `schwimmbad
   <https://github.com/adrn/schwimmbad>`_ project.
   There was a longstanding issue with memory leaks and random crashes of the
   emcee implementation of the ``MPIPool`` that have been fixed in schwimmbad.
   schwimmbad also supports several other ``pool`` interfaces that can be used
   for parallel sampling.
   See the :ref:`parallel` tutorial for more details.

3. The ``PTSampler`` has been removed and forked to the `ptemcee
   <https://github.com/willvousden/ptemcee>`_ project.
   The existing implementation had been gathering dust and there aren't enough
   resources to maintain the sampler within the emcee project.
