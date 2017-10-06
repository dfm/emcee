#!/usr/bin/env python
"""
Run this example with:
mpirun -np 2 python examples/loadbalance.py

Author: Manodeep Sinha. Jan 8, 2014
    - part of the load-balancing implementation for emcee.

"""

from __future__ import print_function

import sys
import time
import pickle
import numpy as np

import emcee
from emcee.utils import MPIPool

seed = 1234567


def sort_on_runtime(pos):
    """
    Given an input list of parameter positions (ndim,nwalkers), this
    function returns a sorted-by-runtime (highest to lowest runtime)
    version of that list

    """
    p = np.atleast_2d(pos)
    idx = (np.argsort(p[:, 0]))[::-1]
    return p[idx], idx


def lnprob(x):
    if x[0] >= 0.0:
        time.sleep(x[0])

    return -0.5*x[0]*x[0]


# The typical run-time for the application
mean_times = [0.5, 1.0]
# The variance in runtime in units of mean_time
variances = [0.1, 0.2, 0.5, 1.0, 2.0]

loadbalancing_options = [False, True]
runtime_sorting_options = [None, sort_on_runtime]

ndim = 1
nwalkers = 496
niters = 5

status_file = 'timings.txt'
f = open(status_file, "w")
pickle_file = 'initial_pos.pkl'

f.write("#####################################################################################################\n")
f.write("##   loadbalance    runtime_sorting   iteration       mean_time     variance      ideal       actual \n")
f.write("#####################################################################################################\n")

for mean_time in mean_times:
    for variance_fac in range(len(variances)):
        first = 0
        variance = variances[variance_fac]*mean_time
        for loadbalance in loadbalancing_options:
            for runtime_sorting_option in runtime_sorting_options:

                # Initialize the MPI-based pool used for parallelization.
                pool = MPIPool(loadbalance=loadbalance)

                if not pool.is_master():
                    # Wait for instructions from the master process.
                    pool.wait()
                    sys.exit(0)

                # Initialize the sampler with the chosen specs.
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, runtime_sortingfn=runtime_sorting_option)

                tstart = time.time()
                print("Before running the iterations. loadbalance = {0}".format(loadbalance))
                if first == 0:
                    # Generate random positions (sleep times for lnprob) for the first time
                    p0 = [(mean_time + variance * np.random.randn(ndim)) for col in range(nwalkers)]
                    pos, prob, rstate = sampler.run_mcmc(p0, 1)
                    pkl_file = open(pickle_file, 'wb')
                    pickle.dump(pos, pkl_file, -1)
                    pickle.dump(prob, pkl_file, -1)
                    pickle.dump(rstate, pkl_file, -1)
                    pkl_file.close()
                else:
                    # use these positions written after the burn-in steps
                    pkl_file = open(pickle_file, 'rb')
                    pos = pickle.load(pkl_file)
                    prob = pickle.load(pkl_file)
                    rstate = pickle.load(pkl_file)
                    pkl_file.close()

                first = 1
                t0 = time.time()
                cumulative_time = 0.0
                ideal_time = pos[pos > 0].sum()/(pool.comm.Get_size()-1)
                for iternum, (pos, prob, rstate) in enumerate(sampler.sample(pos,prob,rstate,iterations=niters,storechain=False)):
                    t1 = time.time()
                    print("Done with iteration {0:2d}. time = {1:8.3f} seconds. perfect scaling  = {2:8.3f} ".format(iternum, t1-t0, ideal_time))

                    if runtime_sorting_option is None:
                        integer_runtime_sort = 0
                    else:
                        integer_runtime_sort = 1
                    f.write(" {0:11b}  {1:14d}       {2:9d}       {3:9.1f}      {4:8.2f}      {5:6.2f}       {6:6.2f}\n".format(loadbalance, integer_runtime_sort, iternum+1, mean_time, variance, ideal_time, t1-t0))
                    f.flush()
                    cumulative_time = cumulative_time + t1-t0
                    t0 = t1
                    # This is how long the next iteration should take.
                    ideal_time = pos[pos > 0].sum()/(pool.comm.Get_size()-1)

                t1 = time.time()
                print("Loadbalancing = {0}, time variance = {1}. Total Time taken = {2:0.2f} seconds (avg = {3:0.3f})".format(loadbalance, variance, cumulative_time, cumulative_time/niters))

        f.write("\n")

f.close()
pool.close()
