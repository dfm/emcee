#!/usr/bin/env python
# encoding: utf-8

"""
 markovpy.py
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

import sys

import numpy as np

from ensemble import EnsembleSampler

def mcfit(lnposteriorfn,p0,args=(),sampler=None,N=1000,burnin=0,outfile=None):
    """
    Fit args using MCMC and return samples from the PDF.
    """
    
    p0 = np.array(p0)
    
    if sampler == None:
        sampler = EnsembleSampler(np.shape(p0)[0],np.shape(p0)[1],lnposteriorfn,postargs=args,outfile=outfile)
    
    pos = p0
    state = None
    
    if burnin > 0:
        print "Running: first burn-in pass"
        pos,prob,state = sampler.run_mcmc(pos,state,burnin/2)
        pos,prob,state = sampler.clustering(pos,prob,state)
        print "Running: second burn-in pass"
        pos,prob,state = sampler.run_mcmc(pos,state,burnin/2)
        pos,prob,state = sampler.clustering(pos,prob,state)
        
        sampler.clear_chain()
    
    print "Running: final Markov chain (%d links)"%(N)
    sampler = EnsembleSampler(np.shape(p0)[0],np.shape(p0)[1],lnposteriorfn,postargs=args,outfile=outfile)
    sampler.run_mcmc(pos,state,N)
    
    frac = np.mean(sampler.acceptance_fraction())
    if frac < 0.1:
        print "Warning: acceptance fraction < 10%"
    else:
        print "Acceptance fraction: %.3f"%frac
    
    return sampler.chain,sampler.probability,frac

