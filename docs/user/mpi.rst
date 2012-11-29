.. _mpi:

.. module:: emcee

Using MPI to distribute the computations
========================================

*Added in version 1.2.0*

The standard implementation of ``emcee`` relies on the ``multiprocessing``
module to parallelize tasks. This works well on a single machine with
multiple cores but it is sometimes useful to distribute the computation
across a larger cluster. To do this, we need to do something a little bit
more sophisticated using the `mpi4py module
<http://mpi4py.scipy.org/docs/usrman/index.html>`_. Below, we'll implement
the `quickstart example </user/quickstart>`_ using MPI but first you'll need
to install the prerequisites.


Prerequisites
-------------

First, if you don't already have it, you'll need to install an MPI
implementation. On Mac OS X, this probably looks something like:

::

    brew install mpich2

(you are using `Homebrew <http://mxcl.github.com/homebrew>`_, aren't you?)
The ``mpi4py`` documentation `gives a few tips
<http://mpi4py.scipy.org/docs/usrman/appendix.html#building-mpi>`_ for this
installation procedure.

Next, you'll need to install ``mpi4py``. The preferred method is probably
something like:

::

    pip install mpi4py

but you can take a look at `the documentation
<http://mpi4py.scipy.org/docs/usrman/install.html>`_ for some other options.


Usage
-----

As of version 1.2.0, ``emcee`` includes an interface to make it easy to use
MPI. The default implementation of parallel ``emcee`` uses the ``map`` method
of a  `multiprocessing.Pool
<http://docs.python.org/2/library/multiprocessing.html#module-multiprocessing.pool>`_
object to compute the likelihoods for the walkers in parallel. The
:class:`EnsembleSampler` object accepts an arbitrary ``pool`` argument in
the constructor. The :class:`utils.MPIPool` object


