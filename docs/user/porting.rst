.. _porting:

Porting your code to emcee3 from earlier versions
=================================================

**emcee3** comes with a pile of new features and an API redesigned from the
ground up to be simultaneously more user-friendly and more powerful.
The main difference is that the probabilistic model is no longer specified
using a function.
Instead, the model is defined as a class

.. code-block:: python

    import emcee
    import numpy as np

    def lnprob(x):
        return -0.5 * np.sum(x ** 2)

    ndim, nwalkers = 10, 100
    p0 = np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(p0, 1000)

.. code-block:: python

    import emcee
    import numpy as np

    class MyModel(emcee.BaseWalker):
        def lnpriorfn(self, x):
            return 0.0
        def lnlikefn(self, x):
            return -0.5 * np.sum(x ** 2)

    ndim, nwalkers = 10, 100
    ensemble = emcee.Ensemble(MyModel, np.random.randn(nwalkers, ndim))
    sampler = emcee.Sampler()
    sampler.run(ensemble, 1000)
