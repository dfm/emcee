#!/usr/bin/env python
# encoding: utf-8
# 
#  runmc.py
#  markovpy
#  
#  Created by Dan F-M on 2010-08-10.
# 

import numpy as np
import pylab as pl

import markovpy as mcmc

def main():
    """Run a Markov chain Monte Carlo to fit a fake linear dataset"""
    np.random.seed()
    
    ptrue = [1.,5.,3.]  # real slope, intercept and scatter
    
    # generate some fake data
    n = 10
    x = 10.*np.random.rand(n)
    err = ptrue[2]*np.random.rand(n)
    y = model(x,ptrue)+np.random.randn(n)*err
    
    # fit the data -- sigma sampled in log
    bounds = [[0.,10.],[0.,10.],[0.,6.]]
    samps,frac = mcmc.mcfit(loglike,bounds,args=(x,y,err))
    
    # rescale sigma samples
    samps[:,2] = np.exp(samps[:,2])
    
    # results
    print "m = %.3f +/- %.3f"%(np.mean(samps[:,0]),np.sqrt(np.var(samps[:,0])))
    print "b = %.3f +/- %.3f"%(np.mean(samps[:,1]),np.sqrt(np.var(samps[:,1])))
    print "sigma = %.3f +/- %.3f"%(np.mean(samps[:,2]),np.sqrt(np.var(samps[:,2])))
    
    # plot the data
    pl.errorbar(x,y,yerr=err,fmt='.k')
    
    # plot this fit
    xt = np.array([min(x),max(x)])
    pl.plot(xt,model(xt,ptrue),'--r',lw=2.)
    
    pl.show()

def loglike(p,x,y,yerr):
    """Likelihood of model p given the data (x,y,yerr) assuming Gaussian uncert."""
    err2 = 2.*(np.exp(p[2])+yerr**2.)
    return np.sum(-(model(x,p)-y)**2./err2-0.5*np.log(np.pi*err2))

def model(x,p):
    """Linear model"""
    return p[0]*x+p[1]

if __name__ == '__main__':
    main()
