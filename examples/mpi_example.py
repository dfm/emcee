"""
Run this example with:
mpirun -np 2 python example_mpi_pool.py

"""


from __future__ import print_function


import numpy as np
import emcee

from emcee.utils import MPIPool


pool = MPIPool(debug=False)
nwalkers = 50
ndim = 10
p0 = np.random.rand(nwalkers, ndim)


def log_prob(p):
    #A trivial Gaussian
    return -0.5 * (p ** 2).sum()


if pool.is_master():
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
    for sample in sampler.sample(p0, iterations=100):
        print(sample[0])
    pool.close()
else:
    pool.wait()
