import numpy as np
from emcee import PTSampler

# Example from PTsampler docs at http://dan.iel.fm/emcee/current/user/pt
# 2 well-separated Gaussians, evidence is known analytically.

# Define the means and standard deviations of our multi-modal likelihood:

# mu1 = [1, 1], mu2 = [-1, -1]
mu1 = np.ones(2)
mu2 = -np.ones(2)

# Width of 0.1 in each dimension, circularly symmetric:
sigma = 0.1
ivar = 1.0/(sigma*sigma)
sigma1inv = np.diag([ivar, ivar])
sigma2inv = np.diag([ivar, ivar])

def logl(x):
    dx1 = x - mu1
    dx2 = x - mu2
    return np.logaddexp(-np.dot(dx1, np.dot(sigma1inv, dx1))/2.0,
                        -np.dot(dx2, np.dot(sigma2inv, dx2))/2.0)

# Use a 2D uniform prior, correctly normalized:
def logp(x):
    xmax = 5.0
    if (abs(x).any() > xmax): 
        logp = -np.inf
    else:
        logp = -2.0*np.log(2.0*xmax)
    return logp

# (Approximate) analytic evidence for two identical Gaussian blobs,
# over a uniform prior [-5:5][-5:5] with density 1/100 in this domain:

log_evidence = np.log(2.0 * 2.0*np.pi*sigma*sigma / 100.0)

# Now we can construct a sampler object that will drive the PTMCMC; arbitrarily,
# we choose to use 20 temperatures (the default is for each temperature to
# increase by a factor of sqrt(2), so the highest temperature will be T=1024,
# resulting in an effective sigmaT=32sigma=3.2, which is about the separation of
# our modes). Let's use 100 walkers in the ensemble at each temperature:

ntemps = 20
nwalkers = 100
ndim = 2

sampler = PTSampler(ntemps, nwalkers, ndim, logl, logp)

# Making the sampling multi-threaded is as simple as adding the threads=Nthreads
# argument to PTSampler. We could have modified the temperature ladder using the
# betas optional argument (which should be an array of beta = 1/T values). The
# pool argument also allows to specify our own pool of worker threads if we
# wanted fine-grained control over the parallelism.

nsteps = 1000

# First, we run the sampler for N/10 burn-in iterations:

print "PT burning in for",nsteps/10,"iterations..."
p0 = np.random.uniform(low=-1.0, high=1.0, size=(ntemps, nwalkers, ndim))
for p, lnprob, lnlike in sampler.sample(p0, iterations=nsteps/10):
    pass
sampler.reset()

# Now we sample for nsteps iterations, recording every 10th sample:

print "PT sampling for",nsteps,"iterations..."
for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
                                           lnlike0=lnlike,
                                           iterations=nsteps, thin=10):
    pass
    
# The resulting samples (nsteps/thin of them) are stored as the sampler.chain 
# property:

assert sampler.chain.shape == (ntemps, nwalkers, nsteps/10, ndim)

# Chain must have shape (ntemps, nwalkers, nsteps, ndim)...

# Zero temperature mean:
# mu0 = np.mean(np.mean(sampler.chain[0,...], axis=0), axis=0)

# Longest autocorrelation length (over any temperature)
# max_acl = np.max(sampler.acor)

# Compute the evidence.
# API notes at http://dan.iel.fm/emcee/current/api/#the-parallel-tempered-ensemble-sampler
 
approximation, uncertainty = sampler.thermodynamic_integration_log_evidence()

# Report!

print "Estimated log evidence = ",approximation,"+/-",uncertainty
print " Analytic log evidence = ",log_evidence

# PT burning in for 100 iterations...
# PT sampling for 1000 iterations...
# Estimated log evidence =  -16.9586413385 +/- 7.29507626509
#  Analytic log evidence =  -6.67931612501

# BUG!
