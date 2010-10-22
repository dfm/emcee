#!/usr/bin/env python
# encoding: utf-8
"""
 runmc.py
 markovpy
 
 Created by Dan F-M on 2010-08-10.


 Copyright 2010 Daniel Foreman-Mackey
  
 This is part of MarkovPy.
 
 MarkovPy is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License version 2 as
 published by the Free Software Foundation.

 MarkovPy is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with MarkovPy.  If not, see <http://www.gnu.org/licenses/>.

"""



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
    bounds = [[0.,10.],[0.,10.],[0.,2.]]
    samps,frac = mcmc.mcfit(loglike,bounds,N=1000,sampler=mcmc.EnsembleSampler(200),args=(x,y,err))
    
    # rescale sigma samples
    samps[:,2] = np.exp(samps[:,2])
    
    pl.figure()
    pl.hist(samps[:,0],100)
    pl.figure()
    pl.hist(samps[:,1],100)
    pl.figure()
    pl.hist(samps[:,2],100)
    pl.figure()
    
    
    # results
    print "m = %.3f +/- %.3f"%(np.mean(samps[:,0]),np.sqrt(np.var(samps[:,0])))
    print "b = %.3f +/- %.3f"%(np.mean(samps[:,1]),np.sqrt(np.var(samps[:,1])))
    print "sigma = %.3f +/- %.3f"%(np.mean(samps[:,2]),np.sqrt(np.var(samps[:,2])))
    
    # plot the data
    pl.errorbar(x,y,yerr=err,fmt='.k')
    
    # plot this fit
    xt = np.array([min(x),max(x)])
    pl.plot(xt,model(xt,ptrue),'--k',lw=2.)
    pl.plot(xt,model(xt,[np.mean(samps[:,0]),np.mean(samps[:,1]),np.mean(samps[:,2])]),'--r',lw=2.)
    
    # blah
    
    # fit the data -- sigma sampled in log
    samps,frac = mcmc.mcfit(loglike,bounds,args=(x,y,err))
    
    # rescale sigma samples
    samps[:,2] = np.exp(samps[:,2])
    
    # results
    print "m = %.3f +/- %.3f"%(np.mean(samps[:,0]),np.sqrt(np.var(samps[:,0])))
    print "b = %.3f +/- %.3f"%(np.mean(samps[:,1]),np.sqrt(np.var(samps[:,1])))
    print "sigma = %.3f +/- %.3f"%(np.mean(samps[:,2]),np.sqrt(np.var(samps[:,2])))
    
    # plot this fit
    xt = np.array([min(x),max(x)])
    pl.plot(xt,model(xt,[np.mean(samps[:,0]),np.mean(samps[:,1]),np.mean(samps[:,2])]),'--b',lw=2.)
    
    
    #end blah
    
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
