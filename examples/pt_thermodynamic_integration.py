import emcee
import matplotlib.pyplot as plt
import numpy as np


def logl(x):
    return -0.5*(np.sum(x**2) + np.log(2*np.pi))


def logp(x):
    a = 5
    if np.abs(x) < a:
        return -np.log(2*a)
    else:
        return -np.inf

ntemps = 100
nwalkers = 100
ndim = 1
betas = np.logspace(0, -3, ntemps)
sampler = emcee.PTSampler(ntemps, nwalkers, ndim, logl=logl, logp=logp,
                          betas=betas)
p0 = np.random.uniform(-1, 1, size=(ntemps, nwalkers, ndim))
out = sampler.run_mcmc(p0, 200)

ln_evidence, ln_error = sampler.thermodynamic_integration_log_evidence()
evidence = np.exp(ln_evidence)
error = evidence * ln_error
print "Evidence={} +/- {}".format(evidence, error)

# Plot of the <ln(Z)> against beta which is integrated over
logls = sampler.lnlikelihood
mean_logls = np.mean(np.mean(logls, axis=1)[:, :], axis=1)
plt.semilogx(betas, mean_logls, "-")
plt.ylabel(r"Mean log-likelihood")
plt.xlabel(r"$\beta = \frac{1}{T}$")
plt.tight_layout()
plt.savefig("betas.png")
plt.show()
