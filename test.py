#!/usr/bin/env python
# encoding: utf-8
# 
#  runmc.py
#  markovpy
#  
#  Created by Dan F-M on 2010-08-10.
# 

# import argparse
import sys

import numpy as np
import pylab as pl

import markovpy as mcmc

def main():
    """Run a Markov chain Monte Carlo"""
    np.random.seed()
    
    n = 10
    ptrue = [1.,5.]
    
    x = 10.*np.random.rand(n)
    err = 3.*np.random.rand(n)
    y = model(x,ptrue)+np.random.randn(n)*err
    # 
    # pl.errorbar(x,y,yerr=err,fmt='.k')
    # xt = np.array([min(x),max(x)])    
    bounds = [[-10.,10.],[-10.,10.],[-10.,10.]]
    samps,frac = mcmc.mcfit(loglike,bounds,args=(x,y,err))
    
    for i in range(len(bounds)):
        pl.figure()
        pl.hist(samps[:,i],50)
    
    # print frac
    # print "m = %.3f +/- %.3f"%(np.mean(samps[:,0]),np.sqrt(np.var(samps[:,0])))
    # print "b = %.3f +/- %.3f"%(np.mean(samps[:,1]),np.sqrt(np.var(samps[:,1])))
    
    # for i in range(50):
    #     ind = np.random.randint(len(samps[:,0]))
    #     pl.plot(xt,model(xt,samps[ind,:]))
    
    # pl.plot(xt,model(xt,ptrue),'--r',lw=2.)
    
    
    pl.show()

def lGauss(x,mu,sig):
    err2 = 2.*sig**2.
    return -(x-mu)**2./err2-0.5*np.log(np.pi*err2)

def loglike(p,x,y,err):
    return lGauss((p[0]+p[1])/p[2],-0.5,1.)+lGauss(p[1],0.,0.1)+lGauss(p[2],0.,0.9)
    
    # err2 = 2.*err**2.
    # return np.sum(-(model(x,p)-y)**2./err2-0.5*np.log(np.pi*err2))

def model(x,p):
    return p[0]*x+p[1]

if __name__ == '__main__':
    sys.exit(main())
