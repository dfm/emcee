#!/usr/bin/env python
# encoding: utf-8
"""
itertest.py

Created by Dan F-M on 2010-11-02.
"""

import numpy as np
import pylab as pl

import markovpy

def main():
    f = lambda x: -np.sum((x)**2)/2
    sampler = markovpy.ensemble.EnsembleSampler(10,15,f)
    
    pos = np.random.randn(15*10).reshape([15,10])
    
    # for k,(i,j) in enumerate(sampler.sample(pos,0.0,iterations=100)):
    #     if k%100 == 0:
    #         print k+1, np.mean(sampler.acceptance_fraction())
    
    sampler.run_mcmc(pos,0.0,1000)
    
    # print j
    samps = sampler.chain
    
    pl.hist(samps[:,9,:].flatten(),100)
    
    pl.figure()
    
    pl.plot(sampler.probability.T)
    pl.show()


if __name__ == '__main__':
    main()

