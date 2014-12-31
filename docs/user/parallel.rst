.. _parallel:

.. module:: emcee

Parallelization
===============

**emcee** supports parallelization out of the box. The algorithmic details are
given in `the paper <http://arxiv.org/abs/1202.3665>`_ but the implementation
is very simple. The parallelization is applied across the walkers in the
ensemble at each step and it must therefore be synchronized after each
iteration. This means that you will really only benefit from this feature when
your probability function is relatively expensive to compute.

The recommended method is to use `IPython's parallel feature
<http://ipython.org/ipython-doc/dev/parallel/>`_ but it's possible to use
other "mappers" like the Python standard library's ``multiprocessing.Pool``.
The only requirement of the mapper is that it exposes a ``map`` method.


Using multiprocessing
---------------------

As mentioned above, it's possible to parallelize your model using the standard
library's ``multiprocessing`` package. Instead, I would recommend the
:class:`pools.InterruptiblePool` that is included with **emcee** because it is
a simple thin wrapper around ``multiprocessing.Pool`` with support for a
keyboard interrupt (``^C``)… you'll thank me later! If we wanted to use this
pool, the final few lines from the example on the front page would become the
following:

.. code-block:: python

    # The ensemble knows about the pool.
    pool = emcee.pools.InterruptiblePool()
    ensemble = emcee.Ensemble(MyModel(), np.random.randn(nwalkers, ndim),
                              pool=pool)

    # These lines are the same.
    sampler = emcee.Sampler()
    sampler.run(ensemble, 1000)

    # Don't forget to close the pool!!
    pool.close()

.. note:: Don't forget to close the pool! It is **your responsibility** as the
          user to close the pool. Otherwise, the Python processes that get
          initialized to run your code won't shut down until your main process
          exits. It's not enough to ``del`` the pool, you have to close it!


Using IPython.parallel
----------------------

`IPython.parallel <http://ipython.org/ipython-doc/dev/parallel/>`_ is a
flexible and powerful framework for running distributed computation in Python.
It works on a single machine with multiple cores in the same way as it does on
a huge compute cluster and in both cases it is very efficient!

To use IPython parallel, make sure that you have a recent version of IPython
installed (`IPython docs <http://ipython.org/>`_) and start up the cluster
by running:

.. code-block:: bash

    ipcluster start

Then, run the following:

.. code-block:: python

    # Connect to the cluster.
    from IPython.parallel import Client
    rc = Client()
    dv = rc.direct_view()

    # Run the imports on the cluster too.
    with dv.sync_imports():
        import emcee
        import numpy

    # Define the model.
    class MyModel(emcee.BaseWalker):
        def lnpriorfn(self, x):
            return 0.0
        def lnlikefn(self, x):
            return -0.5 * numpy.sum(x ** 2)

    # Distribute the model to the nodes of the cluster.
    dv.push(dict(MyModel=MyModel), block=True)

    # Set up the ensemble with the IPython "DirectView" as the pool.
    ndim, nwalkers = 10, 100
    ensemble = emcee.Ensemble(MyModel(), numpy.random.randn(nwalkers, ndim),
                              pool=dv)

    # Run the sampler in the same way as usual.
    sampler = emcee.Sampler()
    sampler.run(ensemble, 1000)

There is a significant overhead incurred when using any of these
parallelization methods so for this simple example, the parallel version is
actually *slower* but this effect will be quickly offset if your probability
function is computationally expensive.

One major benefit of using IPython.parallel is that it can also be used
identically on a cluster with MPI if you have a really big problem. The Python
code would look identical and the only change that you would have to make is
to start the cluster using:

.. code-block:: bash

    ipcluster start --engines=MPI

Take a look at `the documentation
<http://ipython.org/ipython-doc/dev/parallel/>`_ for more details of all of
the features available in IPython.parallel.
