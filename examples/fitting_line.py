#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import math
import time
import emcee

#################################################################################
#######################Some vectorized distributions#############################
############that can be used as priors in vectorized densities###################
#################################################################################


#scipy.stats.norm.logpdf(x, mu, s)
def lnnorm(x, mu, s):
    """Normal distribution with mean `mu` and std `s`. Support (-inf, +inf)."""

    return -0.5 * math.log(2. * math.pi * s ** 2) - (x - mu) ** 2 / (2. * s **
            2.)


#scipy.stats.beta.logpdf(x, alpha, beta)
def lnbeta(x, alpha, beta):
    """\Beta(alpha, beta) - distribution. Support [0, 1]"""

    x_ = np.where((0 < x) & (x < 1), x, 1)

    result1 = -np.log(sp.special.beta(alpha, beta)) + (alpha - 1.) * np.log(x_)\
    + (beta - 1.) * np.log(1. - x_)
    result = np.where((0 < x) & (x < 1), result1, float("-inf"))

    return result


#scipy.stats.uniform.logpdf(x, a, b - a)
def lnunif(x, a, b):
    """Uniform distribution between `a` and `b`."""

    result1 = -np.log(b - a)
    result = np.where((a <= x) & (x <= b), result1, float("-inf"))

    return result


#scipy.stats.lognorm.logpdf(x, i don't know:))
def lnlognorm(x, mu, s):
    """Log-Normal distribution with mean `mu` and std `s`. Support (0,
    +inf)."""

    #going to moments of corresponding normal
    s_ = math.log(s ** 2 / mu ** 2. + 1.)
    mu_ = math.log(mu) - 0.5 * s_

    x_ = np.where(0 < x, x, 1)
    result1 = -np.log(np.sqrt(2. * math.pi * s_) * x_) - (np.log(x_) - mu_)\
    ** 2 / (2. * s_)
    result = np.where(0 < x, result1, float("-inf"))

    return result


#scipy.stats.chi2.logpdf(x, k)
def lnchisq(x, k):
    """Chi-squared distribution with `k` degrees of freedom. Support [0,
    +inf]."""

    x_ = np.where(0 < x, x, 1)
    result1 = (k / 2. - 1.) * np.log(x_) - x_ / 2. - (k / 2.) * math.log(2.) -\
    math.log(sp.special.gamma(k / 2.))

    result = np.where(0 < x, result1, float("-inf"))

    return result


#################################################################################
############################Vectorized densities#################################
#################################################################################


def vmultivariate_gauss(x, means, icov):
    """Vectorized multivariate gaussian density.
    Inputs:
        x [numpy.ndarray] (nwalkers, ndim,) - ensemble of `nwalker` walkers
        (points in parameter space)
        means [numpy.ndarray] (ndim,) - means of distribution,
        icov [numpy.ndarray]  (ndim, ndim,) - inverse covariance matrix.
    Output:
        [numpy.ndarray] (nwalkers,) - ln of probability for walkers in
        `ensemble`.
    """

    x_t = np.array(x).T

    diff = x_t - means[:, np.newaxis]

    return -0.5 * np.einsum('i..., i...', diff, np.dot(icov, diff))


##########################################################################
###########################Linear Model###################################
##########################################################################


def model(walker, x):
    """
    1D linear model y = a * x + b.
    Input:
        x [numpy.ndarray] (ndata,) - vector of predictors,
        walker [numpy.ndarray] (2,) - point (a, b) in parameter space (walker),
    Output:
        [numpy.ndarray] - shape = (length,)
    """

    return walker[0] * x + walker[1]


def lnpost(walker, x, y, sy, mu_a, mu_b, s_a, s_b):
    """Returns ln of posterior probability for one walker. Assuming 1D-linear
    dependence y = a * x + b, gaussian noise and gaussian priors for model
    parameters a & b with means `mu_a`, `mu_b` and variances `sa` and
    `sb`.
    Inputs:
        walker [numpy.ndarray] (2,) - point (a, b) in parameter space (walker),
        x [numpy.ndarray] (ndata,) - vector of predictors,
        y [numpy.ndarray] (ndata,) - vector of responces,
        sy [numpy.ndarray] (ndata,) - vector of uncertainties,
        mu_a [float] (1,) - mean of the prior on slope,
        mu_b [float] (1,) - mean of the prior on intercept,
        s_a [float] (1,) - variance of the prior on slope,
        s_b [float] (1,) - variance of the prior on intercept,


    Outputs:
        [float] (1,) - ln of posterior probability for `walker`.
    """

    #ln of likelyhood
    lnL = (-0.5 * (2 * math.pi * sy ** 2) - (y - model(walker, x)) ** 2 / (2.\
        * sy ** 2)).sum(axis=0)
    #ln of priors
    lnPr = - 0.5 * (2 * math.pi * s_a) - (walker[0] - mu_a) ** 2 / (2. * s_a) -\
    0.5 * (2 * math.pi * s_b) - (walker[1] - mu_b) ** 2 / (2. * s_b)

    result = lnL + lnPr

    return result


def vlnpost(ensemble, x, y, sy, mu_a, mu_b, sa, sb):
    """Returns ln of posterior probability for walkers in ensemble. Assuming
    1D-linear dependence y = a * x + b, gaussian noise and gaussian priors for
    model parameters a & b, with means `mu_a`, `mu_b` and variances `sa` and
    `sb`.

    Inputs:
        ensemble [numpy.ndarray]/[list]/... (nwalkers, 2,) - ensemble of
        #=`nwalker` walkers,
        x [numpy.ndarray] (ndata,) - vector of predictors,
        y [numpy.ndarray] (ndata,) - vector of responces,
        sy [numpy.ndarray] (ndata,) - vector of uncertainties,
        mu_a [float] (1,) - mean of the prior on slope,
        mu_b [float] (1,) - mean of the prior on intercept,
        s_a [float] (1,) - variance of the prior on slope,
        s_b [float] (1,) - variance of the prior on intercept,

    Outputs:
        [numpy.ndarray] (nwalkers,) - ln of posterior probability for walkers
        in `ensemble`.
    """

    ensemble_t = np.array(ensemble).T

    return  lnpost(ensemble_t, x[:, np.newaxis], y[:, np.newaxis], sy[:,
        np.newaxis], mu_a, mu_b, sa, sb)


#linear model with outliers (Hogg et al. arXiv:1008.4686)
def lnpost_outliers(walker, x, y, sy, mu_a, mu_b, mu_Yb, mu_Vb, s_a,
        s_b, s_Yb, s_Vb, alpha, beta):
    """Returns ln of posterior probability for one walker. Assuming 1D-linear
    dependence y = a * x + b, gaussian noise and gaussian noise for outliers.

    a, b - slope and intercept of linear dependence, Pb - prior probability
    that any individual data point is "bad", Yb, Vb - mean and variance of the
    distribution of bad point (in y).

    Used priors:

        Gaussian priors for model parameters a, b & Yb with means `mu_a`, `mu_b` &
        `mu_Yb`, and std `s_a`, `s_b` & `s_Yb`,

        beta prior on Pb with parameters (`alpha`, `beta`),

        lognormal prior on Vb with mean `mu_Vb` and std `s_Vb`.

    Inputs:
        walker [numpy.ndarray] (5,) - point (a, b, Yb, Vb, Pb) in parameter space (walker),
        x [numpy.ndarray] (ndata,) - vector of predictors,
        y [numpy.ndarray] (ndata,) - vector of responces,
        sy [numpy.ndarray] (ndata,) - vector of uncertainties,
        mu_a [float] (1,) - mean of the prior on slope,
        mu_b [float] (1,) - mean of the prior on intercept,
        mu_Yb [float] (1,) - mean of the prior on Yb,
        mu_Vb [float] (1,) - mean of the prior on Vb,
        s_a [float] (1,) - std of the prior on slope a,
        s_b [float] (1,) - std of the prior on intercept b,
        s_Vb [float] (1,) - std of the prior on Vb,
        s_Yb [float] (1,) - std of the prior on Yb,
        alpha, beta - parameters of prior on Pb.

    Outputs:
        [float] (1,) - ln of posterior probability for `walker`.
    """

    #ln of likelyhood - calculate sum of two normals for each data point as sum of to exp's
    lnL = (np.logaddexp(-(y - model(walker, x)) ** 2 / (2. * sy ** 2) +\
        np.log((1. - walker[4]) / np.sqrt(2. * math.pi * sy ** 2)),\
        -(y - walker[2]) ** 2 / (2. * walker[3] + sy ** 2) + np.log(walker[4] /\
            np.sqrt(2. * math.pi * (walker[3] + sy ** 2))))).sum(axis=0)

    #ln of priors
    lnPr = - 0.5 * (2 * math.pi * s_a) - (walker[0] - mu_a) ** 2 / (2. * s_a) -\
        0.5 * (2 * math.pi * s_b) - (walker[1] - mu_b) ** 2 / (2. * s_b) + \
        sp.stats.norm.logpdf(walker[0], mu_a, s_a) +\
        sp.stats.norm.logpdf(walker[1], mu_b, s_b) +\
        sp.stats.norm.logpdf(walker[2], mu_Yb, s_Yb) +\
        lnlognorm(walker[3], mu_Vb, s_Vb) +\
        sp.stats.beta.logpdf(walker[4], alpha, beta)

    result = lnL + lnPr

    return result


#vectorized posterior probability
def vlnpost_outliers(ensemble, x, y, sy, mu_a, mu_b, mu_Yb, mu_Vb, s_a,
        s_b, s_Yb, s_Vb, alpha, beta):
    """Returns ln of posterior probability for walker in ensemble. Assuming
    1D-linear dependence y = a * x + b, gaussian noise and gaussian noise for
    outliers.

    a, b - slope and intercept of linear dependence, Pb - prior probability
    that any individual data point is "bad", Yb, Vb - mean and variance of the
    distribution of bad point (in y).

    Used priors:

        Gaussian priors for model parameters a, b & Yb with means `mu_a`, `mu_b` &
        `mu_Yb`, and std `s_a`, `s_b` & `s_Yb`,

        beta prior on Pb with parameters (`alpha`, `beta`),

        lognormal prior on Vb with mean `mu_Vb` and std `s_Vb`.

    Inputs:
        ensemble [numpy.ndarray] (nwalkers, 5,) - #nwalkers points (a, b, Yb,
        Vb, Pb) in parameter space (#nwalkers walkers),

    Outputs:
        [numpy.ndarray] (nwalkers,) - ln of posterior probability for walkers
        in `enseble`.
    """

    ensemble_t = np.array(ensemble).T

    return lnpost_outliers(ensemble_t, x[:, np.newaxis], y[:, np.newaxis],
            sy[:, np.newaxis], mu_a, mu_b, mu_Yb, mu_Vb, s_a, s_b, s_Yb, s_Vb,
            alpha, beta)


ndim = 5
nwalkers = 500

p0 = [np.random.rand(ndim) for i in range(nwalkers)]

#loading data
id_, x, y, sy, sx, ro_xy = np.loadtxt('outliers.data', unpack=True)

#specifying prior's moments
mu_a = 2.
mu_b = 40.
mu_Yb = np.mean(y)
mu_Vb = 200.

#for beta-prior on Pb
alpha = 8
beta = 2

s_a = 4.
s_b = 80.
s_Yb = 1000.
s_Vb = 400.

#arguments to callable posterior
args = (x, y, sy, mu_a, mu_b, mu_Yb, mu_Vb, s_a, s_b, s_Yb, s_Vb, alpha, beta)

#initializing samplers
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost_outliers, args=args)
poolsampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost_outliers, args=args, threads=10)
vecsampler = emcee.EnsembleSampler(nwalkers, ndim, vlnpost_outliers, args=args, bcast=True)

#burn in
pos, prob, state = sampler.run_mcmc(p0, 100)
ppos, pprob, pstate = poolsampler.run_mcmc(p0, 100)
vpos, vprob, vstate = vecsampler.run_mcmc(p0, 100)

sampler.reset()
poolsampler.reset()
vecsampler.reset()

##################################################################
##############Timing for fitting outliers model###################
##################################################################

t1 = time.time()
sampler.run_mcmc(pos, 1000)
dt = time.time() - t1
print "time : " + str(dt)
#~400s on mobile i3 Sandy Bridge

t1 = time.time()
poolsampler.run_mcmc(ppos, 1000)
dt = time.time() - t1
print "threads=10 time : " + str(dt)
#~200s

t1 = time.time()
vecsampler.run_mcmc(vpos, 1000)
dt = time.time() - t1
print "bcast=True time : " + str(dt)
#~7s

try:
    import matplotlib.pyplot as pl
except ImportError:
    print("Try installing matplotlib to generate some sweet plots...")
else:
    H, xedges, yedges = np.histogram2d(vecsampler.flatchain[:, 0],
            vecsampler.flatchain[:, 1], bins=(128, 128))
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    pl.imshow(H, interpolation='nearest', extent=extent, aspect="auto")
