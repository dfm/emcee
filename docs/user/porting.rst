.. _porting:

.. module:: emcee

Porting your code to emcee3 from earlier versions
=================================================

**emcee3** comes with a pile of new features and an API redesigned from the
ground up to be simultaneously more user-friendly and more powerful.


Why switch?
-----------

I implemented emcee3 from scratch with a completely re-thought API using
everything that I learned from answering support requests and data-analysis
questions.
emcee3 now supports nearly all of the earlier features (with the notable
exception of the parallel tempering sampler but that should be added soon) and
a lot of new features that should make everyone's life better!
This isn't the place to list all of the new features but here are some of the
most important ones:

1. First class support for alternative moves and schedules of moves. This
   means that you can sample using one or more of the many included proposals
   (including the original stretch move but also a new walk move, differential
   evolution, and others) or implement your own.
2.


A complete example
------------------

The main difference between older versions of emcee and emcee3 is that the
probabilistic model is no longer specified as a function.
Instead, the model is defined as a class that should probably inherit from
:class:`BaseWalker` or otherwise implement the full walker API.
To get started, let's take the sample code from the front page and show how
you would update it to work with emcee3.
From the front page of the original docs, here is a full example showing you
how to sample from a multidimensional Gaussian:

.. code-block:: python

    # This code is for emcee versions earlier than 3.
    import emcee
    import numpy as np

    def lnprob(x, var):
        return -0.5 * np.sum(x ** 2 / var)

    ndim, nwalkers = 10, 100
    p0 = np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[0.5])
    sampler.run_mcmc(p0, 1000)
    samples = sampler.chain

In emcee3, this example becomes:

.. code-block:: python

    # This is emcee >= 3 code.
    import emcee
    import numpy as np

    class MyModel(emcee.BaseWalker):
        def __init__(self, var):
            self.var = var
        def lnpriorfn(self, x):
            return 0.0
        def lnlikefn(self, x):
            return -0.5 * np.sum(x ** 2 / self.var)

    ndim, nwalkers = 10, 100
    ensemble = emcee.Ensemble(MyModel(0.5), np.random.randn(nwalkers, ndim))
    sampler = emcee.Sampler()
    sampler.run(ensemble, 1000)
    samples = sampler.get_coords()

As mentioned above, the main difference is that the model is implemented as a
class.
The code for the ``lnlikefn`` method is *identical* to the ``lnprob`` function
defined previously but the walker must now also implement a ``lnpriorfn`` that
is responsible for evaluating the logarithm of the prior function.

After the differences in model specification, the only other major difference
is that the ensemble state is stored in a specific object instead of a set of
numpy arrays.
To construct a new ensemble with the specified model ``MyModel`` at a set of
random coordinates, you run:

.. code-block:: python

    ensemble = emcee.Ensemble(MyModel(0.5), np.random.randn(nwalkers, ndim))

As with previous versions of emcee, it is the user's responsibility to specify
the initial coordinates as a numpy matrix with dimensions ``(nwalkers,
ndim)``.
In general, the best initial condition is to have the walkers distributed in a
"small" clump around your current best guess of the solution.

Results
+++++++

To access the sampling results, the best practice is to use the methods
defined on :class:`Sampler`.
Access to the chain of samples is provided by the :func:`Sampler.get_coords`
method and the result will have the shape: ``(nsteps, nwalkers, ndim)``.
You can also get the "flattened" chain by calling
``sampler.get_coords(flat=True)``.
Other keyword arguments can be used to thin the chain or discard burn-in
samples.
Similar methods are also available for the ln-prior, ln-likelihood, and
ln-probability chains.

.. note:: The shape of the chain returned by emcee3 is different from previous
          versions. In emcee3, the chain is a *list of ensembles* so it has
          the shape ``(nsteps, nwalkers, ndim)``.
