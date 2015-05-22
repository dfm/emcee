import numpy as np
from emcee import PTSampler, default_beta_ladder

try:
    import matplotlib.pyplot as plt
except:
    plt = None

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
    if (np.any(np.abs(x) > xmax)): 
        logp = -np.inf
    else:
        logp = -2.0*np.log(2.0*xmax)
    return logp

# (Approximate) analytic evidence for two identical Gaussian blobs,
# over a uniform prior [-5:5][-5:5] with density 1/100 in this domain:

log_evidence = np.log(2.0 * 2.0*np.pi*sigma*sigma / 100.0)

# If we were just trying to sample from this distribution, we would probably
# choose to use 7 temperatures.  The default temperature step factor in two
# dimensions is 7 (see default_beta_ladder), so the highest temperature would
# then be T=7^6=120000, resulting in an effective sigmaT=350*sigma=35, which is
# much greater than the separation of our modes).  However, this spacing is too
# wide to get a good evidence computation, since that is an integral over
# temperature.  So, we will use our own set of 30 betas, distributed uniformly
# in log(beta) between 1 and 1/10^6.

# Let's use 100 walkers in the ensemble at each temperature:

ntemps = 30
nwalkers = 100
ndim = 2
betas = np.exp(np.linspace(0.0, -np.log(1e6), ntemps))
# betas should be decreasing, not increasing
betas = betas[::-1]

sampler = PTSampler(ntemps, nwalkers, ndim, logl, logp, betas=betas)

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

print 'Temperature swap acceptance rates are '
for b, rate in zip(sampler.betas, sampler.tswap_acceptance_fraction):
    print 'T = ', 1.0/b, ' accept = ', rate
print

# I recommend that you *always* plot the TI integrand like this---you can see
# immediately whether you need a denser spacing in T, or whether your
# high-temperature limit is high enough, etc.  Good sampling is not necessarily
# an indicator of good convergence in the TI integral!
if plt is not None:
    # Print a plot of the TI integrand:
    mean_logls = np.mean(sampler.lnlikelihood.reshape((ntemps, -1)), axis=1)
    betas = sampler.betas
    plt.plot(betas, betas*mean_logls) # \int d\beta <logl> = \int d\ln\beta \beta <logl>
    plt.xscale('log')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\beta \left\langle \ln L \right\rangle_\beta$')
    plt.title('Thermodynamic Integration Integrand')
    plt.show()

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
