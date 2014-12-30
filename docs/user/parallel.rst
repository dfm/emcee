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

Then, update the code from the front page example as follows and save it in a
file called something like ``model.py``. Note: this file should be in the
directory as where you started the ``ipcluster`` or it should be otherwise
importable from the cluster process.

.. code-block:: python

    # file: model.py
    import emcee
    import numpy as np

    class MyModel(emcee.BaseWalker):
        def lnpriorfn(self, x):
            return 0.0
        def lnlikefn(self, x):
            return -0.5 * np.sum(x ** 2)

Then, run the following:

.. code-block:: python

    import emcee
    import numpy as np
    from model import MyModel

    # Connect to the cluster.
    from IPython.parallel import Client
    rc = Client()
    pool = rc.load_balanced_view()

    # Everything below here is the same.
    ndim, nwalkers = 10, 100
    ensemble = emcee.Ensemble(MyModel(), np.random.randn(nwalkers, ndim),
                              pool=pool)
    sampler = emcee.Sampler()
    sampler.run(ensemble, 1000)
