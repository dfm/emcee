import os
import sys
import time
from multiprocessing import Pool

import h5py
import matplotlib.pyplot as pl
import numpy as np

import emcee

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))

# import acor


def lnprobfn(p, icov):
    return -0.5 * np.dot(p, np.dot(icov, p))


def random_cov(ndim, dof=1):
    v = np.random.randn(ndim * (ndim + dof)).reshape((ndim + dof, ndim))
    return sum([np.outer(v[i], v[i]) for i in range(ndim + dof)]) / (
        ndim + dof
    )


_rngs = {}


def _worker(args):
    i, outfn, nsteps = args

    pid = os.getpid()
    _random = _rngs.get(
        pid, np.random.RandomState(int(int(pid) + time.time()))
    )
    _rngs[pid] = _random

    ndim = int(np.ceil(2 ** (7 * _random.rand())))
    nwalkers = 2 * ndim + 2
    # nwalkers += nwalkers % 2
    print(ndim, nwalkers)

    cov = random_cov(ndim)
    icov = np.linalg.inv(cov)

    ens_samp = emcee.EnsembleSampler(nwalkers, ndim, lnprobfn, args=[icov])
    ens_samp.random_state = _random.get_state()
    pos, lnprob, state = ens_samp.run_mcmc(
        np.random.randn(nwalkers * ndim).reshape([nwalkers, ndim]), nsteps
    )

    proposal = np.diag(cov.diagonal())
    mh_samp = emcee.MHSampler(proposal, ndim, lnprobfn, args=[icov])
    mh_samp.random_state = state
    mh_samp.run_mcmc(np.random.randn(ndim), nsteps)

    f = h5py.File(outfn)
    f["data"][i, :] = np.array(
        [ndim, np.mean(ens_samp.acor), np.mean(mh_samp.acor)]
    )
    f.close()


def oned():
    nsteps = 10000
    niter = 10
    nthreads = 2

    outfn = os.path.join(os.path.split(__file__)[0], "gauss_scaling.h5")
    print(outfn)
    f = h5py.File(outfn, "w")
    f.create_dataset("data", (niter, 3), "f")
    f.close()

    pool = Pool(nthreads)
    pool.map(_worker, [(i, outfn, nsteps) for i in range(niter)])

    f = h5py.File(outfn)
    data = f["data"][...]
    f.close()

    pl.clf()
    pl.plot(data[:, 0], data[:, 1], "ks", alpha=0.5)
    pl.plot(data[:, 0], data[:, 2], ".k", alpha=0.5)

    pl.savefig(os.path.join(os.path.split(__file__)[0], "gauss_scaling.png"))


if __name__ == "__main__":
    oned()
