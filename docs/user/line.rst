.. _line:
.. module:: emcee

Fitting a Line to Noisy Data
============================

If you're reading this right now then you're probably interested in using
emcee to fit a model to some noisy data.
On this page, I'll demonstrate how you might do this in the simplest
non-trivial model that I could think of: fitting a line to data when you
don't believe the error bars on your data.

The generative probabilistic model
----------------------------------

When you approach a new problem, the first step is generally to write down the
*likelihood function* (the probability of a dataset given the model
parameters).
This is equivalent to describing the generative procedure for the data.
In this case, we're going to consider a linear model where the quoted
uncertainties are underestimated by a constant fractional amount.
You can generate a synthetic dataset from this model

.. code-block:: python

    import numpy as np

    # True parameters.
    m_true = -0.2594
    b_true = 3.294
    f_true = 0.534

    N = 100
    x = np.sort(10*np.random.rand(N))
    y_err = 0.5+np.random.rand(N)
    y = m_true*x+b_true
    y += (f_true*y) * np.random.randn(N)
    y += y_err * np.random.randn(N)
