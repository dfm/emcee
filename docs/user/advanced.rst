.. _advanced:

.. module:: emcee

Advanced Patterns
=================

``emcee`` is generally pretty simple but it has a few key features that make
the usage easier in real problems. Here are a few examples of things that
you might find useful.


Incrementally saving progress
-----------------------------

It is often useful to incrementally save the state of the chain to a file.
This makes it easier to monitor the chain's progress and it makes things a
little less disastrous if your code/computer crashes somewhere in the middle
of an expensive MCMC run. If you just want to append the walker positions to
the end of a file, you could do something like:

.. code-block:: python

    f = open("chain.dat", "w")
    f.close()

    for result in sampler.sample(pos0, iterations=500, storechain=False):
        position = result[0]
        f = open("chain.dat", "a")
        for k in range(position.shape[0]):
            f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str,position[k]))))
        f.close()


Printing the sampler's progress
-------------------------------

You might want to monitor the progress of the sampler in your terminal while it
runs.  There are several modules out there that can help you make shiny progress
bars (e.g., are `progressbar <https://pypi.python.org/pypi/progressbar>`_ and
`clint <http://pypi.python.org/pypi/clint/>`_), but it's straightforward to
implement a simple progress counter yourself.

The solution here is very similar to the incremental saving snippet.  For
example, to display the current percentage:

.. code-block:: python

    nsteps = 5000
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
        if (i+1) % 100 == 0:
            print("{0:5.1%}".format(float(i) / nsteps))

Or, to display a rudimentary progress bar that updates iteself on a single line:

.. code-block:: python

    import sys

    nsteps = 5000
    width = 30
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
        n = int((width+1) * float(i) / nsteps)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
    sys.stdout.write("\n")

Multiprocessing
---------------

In principle, running ``emcee`` in parallel is as simple instantiating an
:class:`EnsembleSampler` object with the ``threads`` argument set to an
integer greater than 1:

.. code-block:: python

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpostfn, threads=15)

In practice, the parallelization is implemented using the built in Python
`multiprocessing <http://docs.python.org/library/multiprocessing.html>`_
module. With this comes a few constraints. In particular, both ``lnpostfn``
and ``args`` must be `pickleable
<http://docs.python.org/library/pickle.html#what-can-be-pickled-and-unpickled>`_.
The exceptions thrown while using ``multiprocessing`` can be quite cryptic
and even though we've tried to make this feature as user-friendly as possible,
it can sometimes cause some headaches. One useful debugging tactic is to
try running with 1 thread if your processes start to crash. This will
generally provide much more illuminating error messages than in the parallel
case. Note that the parallelized :class:`EnsembleSampler` object is not
pickleable. Therefore, if it (or an object that contains it) is passed to
``lnpostfn`` when multiprocessing is turned on, the code will fail.

It is also important to note that the ``multiprocessing`` module works by
spawning a large number of new ``python`` processes and running the code in
isolation within those processes. This means that there is a significant
amount of overhead involved at each step of the parallelization process.
With this in mind, it is not surprising that running a simple problem like
the :ref:`quickstart example <quickstart>` in parallel will run much slower
than the equivalent serial code. If your log-probability function takes
a significant amount of time (> 1 second or so) to compute then using the
parallel sampler actually provides significant speed gains.


.. _blobs:

Arbitrary metadata blobs
------------------------

*Added in version 1.1.0*

Imagine that your log-probability function involves an extremely
computationally expensive numerical simulation starting from initial
conditions parameterized by the position of the walker in parameter space.
Then you have to compare the results of your simulation by projecting into
data space (predicting you data) and computing something like a chi-squared
scalar in this space. After you run MCMC, you might want to visualize
the draws from your probability function in data space by over-plotting
samples on your data points. It is obviously unreasonable to recompute
all the simulations for all the initial conditions that you want to display
as a part of your post-processing—especially since you already computed all
of them before! Instead, it would be ideal to be able to store realizations
associated with each step in the MCMC and then just display those after the
fact. This is possible using the "arbitrary blob" pattern.

To use ``blobs``, you just need to modify your log-probability function to
return a second argument (this can be any arbitrary Python object). Then,
the sampler object will have an attribute (called
:attr:`EnsembleSampler.blobs`) that is a list (of length ``niterations``)
of lists (of length ``nwalkers``) containing all the accepted ``blobs``
associated with the walker positions in :attr:`EnsembleSampler.chain`.

As an absolutely trivial example, let's say that we wanted to store the
sum of cubes of the input parameters as a string at each position in the
chain. To do this we could simply sample a function like:

.. code-block:: python

    def lnprobfn(p):
        return -0.5 * np.sum(p ** 2), str(np.sum(p ** 3))

It is important to note that by returning two values from our log-probability
function, we also change the output of :func:`EnsembleSampler.sample` and
:func:`EnsembleSampler.run_mcmc` to return 4 values (position, probability,
random number generator state and blobs) instead of just the first three.

.. _mpi:

Using MPI to distribute the computations
----------------------------------------

*Added in version 1.2.0*

The standard implementation of ``emcee`` relies on the ``multiprocessing``
module to parallelize tasks. This works well on a single machine with
multiple cores but it is sometimes useful to distribute the computation
across a larger cluster. To do this, we need to do something a little bit
more sophisticated using the `mpi4py module
<http://mpi4py.scipy.org/docs/usrman/index.html>`_. Below, we'll implement
an example similar to the `quickstart <../quickstart>`_ using MPI but
first you'll need to `install mpi4py
<http://mpi4py.scipy.org/docs/usrman/install.html>`_.

The :class:`utils.MPIPool` object provides most of the needed functionality
so we'll start by importing that and the other needed modules:

.. code-block:: python

    import sys
    import numpy as np
    import emcee
    from emcee.utils import MPIPool

This time, we'll just sample a simple isotropic Gaussian (remember that the
``emcee`` algorithm *doesn't care about covariances between parameters
because it is affine-invariant*):

.. code-block:: python

    ndim = 50
    nwalkers = 250
    p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]

    def lnprob(x):
        return -0.5 * np.sum(x ** 2)

Now, this is where things start to change:

.. code-block:: python

    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

First, we're initializing the pool object and then---if the process isn't
running as master---we wait for instructions and then exit. Then, we can
set up the sampler providing this pool object to do the parallelization:

.. code-block:: python

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

and then run and analyse as usual. The key here is that only the master
chain should *actually* directly interact with the sampler and the other
processes should only wait for instructions.

*Note*: don't forget to close the pool if you don't want the processes to
hang forever:

.. code-block:: python

    pool.close()

The full source code for this example is available `on Github
<https://github.com/dfm/emcee/blob/master/examples/mpi.py>`_.

If we save this script to the file ``mpi.py``, we can then run this example
with the command:

.. code-block:: bash

    mpirun -np 2 python mpi.py

for local testing.

.. _loadbalance:

Loadbalancing in parallel runs
------------------------------

*Added in version 2.1.0*

When ``emcee`` is being used in a multi-processing mode (``multiprocessing`` or
``mpi4py``), the parameters need to distributed evenly over all the available
cores. ``emcee`` uses a ``map`` function to distribute the jobs over the available
cores. In case of ``multiprocessing``, the ``map`` function is in-built and
dynamically schedules the tasks. In order to get a similar dynamic
scheduling in ``map`` when using :class:`utils.MPIPool` , use the following
invocation:

.. code-block:: python

    pool = MPIPool(loadbalance=True)


By default, ``loadbalance`` is set to ``False``. If your jobs have a lot of
variance in run-time, then setting the ``loadbalance`` option will improve
the overall run-time.

If your problem is such that the runtime for each invocation of the
log-probability function scales with one/some of the parameters, then you can
improve load-balancing even further. By sorting the jobs in decreasing order
of (expected) run-time, the longest jobs get run simultaneously and you only
have the wait for the duration of the longest job. In the following example,
the first parameter strongly determines the run-time -- larger the first
parameter, the longer the runtime. The ``sort_on_runtime`` returns the
re-ordered list and the corresponding index.

.. code-block:: python

    def sort_on_runtime(pos):
        p = np.atleast_2d(pos)
        idx = np.argsort(p[:, 0])[::-1]
        return p[idx], idx

In order to use this function, you will have to instantiate an
:class:`EnsembleSampler` object with:

.. code-block:: python

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool,
                                    runtime_sortingfn=sort_on_runtime)


Such a ``sort_on_runtime`` can be applied to both ``multiprocessing``
and ``mpi4py`` invocations for ``emcee``. You can see a benchmarking
routine using the ``mpi4py`` module `on Github
<https://github.com/dfm/emcee/blob/master/examples/loadbalance.py>`_.
