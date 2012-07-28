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

::

    f = open("chain.dat", "w")
    f.close()

    for result in sampler.sample(pos0, iterations=500, storechain=False):
        position = result[0]
        f = open("chain.dat", "a")
        for k in range(position.shape[0]):
            f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k])))
        f.close()


Multiprocessing
---------------

In principle, running ``emcee`` in parallel is as simple instantiating an
:class:`EnsembleSampler` object with the ``threads`` argument set to an
integer greater than 1:

::

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobfn, threads=15)

In practice, the parallelization is implemented using the built in Python
`multiprocessing <http://docs.python.org/library/multiprocessing.html>`_
module. With this comes a few constraints. In particular, both ``lnprobfn``
and ``args`` must be `pickleable
<http://docs.python.org/library/pickle.html#what-can-be-pickled-and-unpickled>`_.
The exceptions thrown while using ``multiprocessing`` can be quite cryptic
and even though we've tried to make this feature as user-friendly as possible,
it can sometimes cause some headaches. One useful debugging tactic is to
try running with 1 thread if your processes start to crash. This will
generally provide much more illuminating error messages than in the parallel
case.

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

*(New in version 1.1.0)*

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

::

    def lnprobfn(p):
        return -0.5 * np.sum(p ** 2), str(np.sum(p ** 3))

It is important to note that by returning two values from our log-probability
function, we also change the output of :func:`EnsembleSampler.sample` and
:func:`EnsembleSampler.run_mcmc` to return 4 values (position, probability,
random number generator state and blobs) instead of just the first three.
