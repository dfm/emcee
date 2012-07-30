import os
import sys

import numpy as np
import matplotlib.pyplot as pl

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
import emcee

# import acor


def lnprobfn(p, icov):
    return -0.5 * np.dot(p, np.dot(icov, p))


def random_cov(ndim, dof=1):
    v = np.random.randn(ndim * (ndim + dof)).reshape((ndim + dof, ndim))
    return (sum([np.outer(v[i], v[i]) for i in range(ndim + dof)])
            / (ndim + dof))


def oned():
    nsteps = 5000
    ens_acor = []
    mh_acor = []
    dims = []

    for i in range(200):
        ndim = int(np.ceil(2 ** (6 * np.random.rand())))
        dims.append(ndim)
        nwalkers = 2 * ndim + 2
        # nwalkers += nwalkers % 2
        print ndim, nwalkers

        cov = random_cov(ndim)
        icov = np.linalg.inv(cov)

        ens_samp = emcee.EnsembleSampler(nwalkers, ndim, lnprobfn,
                args=[icov])
        ens_samp.run_mcmc(np.random.randn(nwalkers * ndim).reshape(
            [nwalkers, ndim]), nsteps)
        ens_acor.append(np.mean(ens_samp.acor))

        proposal = np.diag(cov.diagonal())
        mh_samp = emcee.MHSampler(proposal, ndim, lnprobfn,
                args=[icov])
        mh_samp.run_mcmc(np.random.randn(ndim), nsteps)
        mh_acor.append(np.mean(mh_samp.acor))

        print "\t", ens_acor[-1], mh_acor[-1]

        pl.clf()
        pl.plot(dims, ens_acor, "ks", alpha=0.5)
        pl.plot(dims, mh_acor, ".k", alpha=0.5)

        pl.savefig("oned.png")


if __name__ == "__main__":
    oned()
