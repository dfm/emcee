
.. module:: emcee

**Note:** This tutorial was generated from an IPython notebook that can be
downloaded `here <../../_static/notebooks/parallel.ipynb>`_.

.. _parallel:


Parallelization
===============

.. note:: Some builds of NumPy (including the version included with Anaconda) will automatically parallelize some operations using something like the MKL linear algebra. This can cause problems when used with the parallelization methods described here so it can be good to turn that off (by setting the environment variable ``OMP_NUM_THREADS=1``, for example).

.. code:: python

    import os
    os.environ["OMP_NUM_THREADS"] = "1"

With emcee, it’s easy to make use of multiple CPUs to speed up slow
sampling. There will always be some computational overhead introduced by
parallelization so it will only be beneficial in the case where the
model is expensive, but this is often true for real research problems.
All parallelization techniques are accessed using the ``pool`` keyword
argument in the :class:`EnsembleSampler` class but, depending on your
system and your model, there are a few pool options that you can choose
from. In general, a ``pool`` is any Python object with a ``map`` method
that can be used to apply a function to a list of numpy arrays. Below,
we will discuss a few options.

This tutorial was executed with the following version of emcee:

.. code:: python

    import emcee
    print(emcee.__version__)


.. parsed-literal::

    /Users/dforeman/anaconda/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


.. parsed-literal::

    3.0.0.dev0


In all of the following examples, we’ll test the code with the following
convoluted model:

.. code:: python

    import time
    import numpy as np
    
    def log_prob(theta):
        t = time.time() + np.random.uniform(0.005, 0.008)
        while True:
            if time.time() >= t:
                break
        return -0.5*np.sum(theta**2)

This probability function will randomly sleep for a fraction of a second
every time it is called. This is meant to emulate a more realistic
situation where the model is computationally expensive to compute.

To start, let’s sample the usual (serial) way:

.. code:: python

    np.random.seed(42)
    initial = np.random.randn(32, 5)
    nwalkers, ndim = initial.shape
    nsteps = 100
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    start = time.time()
    sampler.run_mcmc(initial, nsteps, progress=True)
    end = time.time()
    serial_time = end - start
    print("Serial took {0:.1f} seconds".format(serial_time))


.. parsed-literal::

    100%|██████████| 100/100 [00:21<00:00,  4.70it/s]

.. parsed-literal::

    Serial took 21.3 seconds


.. parsed-literal::

    


Multiprocessing
---------------

The simplest method of parallelizing emcee is to use the
`multiprocessing module from the standard
library <https://docs.python.org/3/library/multiprocessing.html>`__. To
parallelize the above sampling, you could update the code as follows:

.. code:: python

    from multiprocessing import Pool
    
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        start = time.time()
        sampler.run_mcmc(initial, nsteps, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        print("{0:.1f} times faster than serial".format(serial_time / multi_time))


.. parsed-literal::

    100%|██████████| 100/100 [00:06<00:00, 16.05it/s]

.. parsed-literal::

    Multiprocessing took 6.2 seconds
    3.4 times faster than serial


.. parsed-literal::

    


I have 4 cores on the machine where this is being tested:

.. code:: python

    from multiprocessing import cpu_count
    ncpu = cpu_count()
    print("{0} CPUs".format(ncpu))


.. parsed-literal::

    4 CPUs


We don’t quite get the factor of 4 runtime decrease that you might
expect because there is some overhead in the parallelization, but we’re
getting pretty close with this example and this will get even closer for
more expensive models.

MPI
---

Multiprocessing can only be used for distributing calculations across
processors on one machine. If you want to take advantage of a bigger
cluster, you’ll need to use MPI. In that case, you need to execute the
code using the ``mpiexec`` executable, so this demo is slightly more
convoluted. For this example, we’ll write the code to a file called
``script.py`` and then execute it using MPI, but when you really use the
MPI pool, you’ll probably just want to edit the script directly. To run
this example, you’ll first need to install `the schwimmbad
library <https://github.com/adrn/schwimmbad>`__ because emcee no longer
includes its own ``MPIPool``.

.. code:: python

    with open("script.py", "w") as f:
        f.write("""
    import sys
    import time
    import emcee
    import numpy as np
    from schwimmbad import MPIPool
    
    def log_prob(theta):
        t = time.time() + np.random.uniform(0.005, 0.008)
        while True:
            if time.time() >= t:
                break
        return -0.5*np.sum(theta**2)
    
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
            
        np.random.seed(42)
        initial = np.random.randn(32, 5)
        nwalkers, ndim = initial.shape
        nsteps = 100
    
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        start = time.time()
        sampler.run_mcmc(initial, nsteps)
        end = time.time()
        print(end - start)
    """)
    
    mpi_time = !mpiexec -n {ncpu} python script.py
    mpi_time = float(mpi_time[0])
    print("MPI took {0:.1f} seconds".format(mpi_time))
    print("{0:.1f} times faster than serial".format(serial_time / mpi_time))


.. parsed-literal::

    MPI took 8.3 seconds
    2.6 times faster than serial


There is often more overhead introduced by MPI than multiprocessing so
we get less of a gain this time. That being said, MPI is much more
flexible and it can be used to scale to huge systems.

Pickling, data transfer & arguments
-----------------------------------

All parallel Python implementations work by spinning up multiple
``python`` processes with identical environments then and passing
information between the processes using ``pickle``. This means that the
probability function `must be
picklable <https://docs.python.org/3/library/pickle.html#pickle-picklable>`__.

Some users might hit issues when they use ``args`` to pass data to their
model. These args must be pickled and passed every time the model is
called. This can be a problem if you have a large dataset, as you can
see here:

.. code:: python

    def log_prob_data(theta, data):
        a = data[0]  # Use the data somehow...
        t = time.time() + np.random.uniform(0.005, 0.008)
        while True:
            if time.time() >= t:
                break
        return -0.5*np.sum(theta**2)
    
    data = np.random.randn(5000, 200)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_data, args=(data,))
    start = time.time()
    sampler.run_mcmc(initial, nsteps, progress=True)
    end = time.time()
    serial_data_time = end - start
    print("Serial took {0:.1f} seconds".format(serial_data_time))


.. parsed-literal::

    100%|██████████| 100/100 [00:21<00:00,  4.71it/s]

.. parsed-literal::

    Serial took 21.3 seconds


.. parsed-literal::

    


We basically get no change in performance when we include the ``data``
argument here. Now let’s try including this naively using
multiprocessing:

.. code:: python

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_data, pool=pool, args=(data,))
        start = time.time()
        sampler.run_mcmc(initial, nsteps, progress=True)
        end = time.time()
        multi_data_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_data_time))
        print("{0:.1f} times faster(?) than serial".format(serial_data_time / multi_data_time))


.. parsed-literal::

    100%|██████████| 100/100 [01:17<00:00,  1.32it/s]

.. parsed-literal::

    Multiprocessing took 77.7 seconds
    0.3 times faster(?) than serial


.. parsed-literal::

    


Brutal.

We can do better than that though. It’s a bit ugly, but if we just make
``data`` a global variable and use that variable within the model
calculation, then we take no hit at all.

.. code:: python

    def log_prob_data_global(theta):
        a = data[0]  # Use the data somehow...
        t = time.time() + np.random.uniform(0.005, 0.008)
        while True:
            if time.time() >= t:
                break
        return -0.5*np.sum(theta**2)
    
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_data_global, pool=pool)
        start = time.time()
        sampler.run_mcmc(initial, nsteps, progress=True)
        end = time.time()
        multi_data_global_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_data_global_time))
        print("{0:.1f} times faster than serial".format(serial_data_time / multi_data_global_time))


.. parsed-literal::

    100%|██████████| 100/100 [00:06<00:00, 16.46it/s]

.. parsed-literal::

    Multiprocessing took 6.2 seconds
    3.4 times faster than serial


.. parsed-literal::

    


That’s better! This works because, in the global variable case, the
dataset is only pickled and passed between processes once (when the pool
is created) instead of once for every model evaluation.

