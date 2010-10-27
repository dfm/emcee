#!/usr/bin/env python
# encoding: utf-8

"""
 mcsampler.py
 markovpy
 
 Created by Dan F-M on 2010-10-18.

 This is a Markov chain Monte Carlo (MCMC) sampler based on:

 Goodman & Weare, Ensemble Samplers With Affine Invariance 
   Comm. App. Math. Comp. Sci., Vol. 5 (2010), No. 1, 65–80


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
from mcsampler import MCSampler

class EnsembleSampler(MCSampler):
    """Ensemble sampling following Goodman & Weare (2009)"""
    def __init__(self,nwalkers,a=2.):
        self.nwalkers = nwalkers
        self.a = a
    
    def sample_pdf(self,logpost,p0,proposal,N,burnin,args,seed=None,outfile=None):
        if seed == None:
            np.random.seed()
        else:
            np.random.seed(seed)
        
        W = self.nwalkers # number of walkers
        
        npars = np.shape(p0)[0]
        old_p,old_prob = [],[]
        chain = []
        
        # initialize walkers
        s = np.shape(p0)
        if s[1] != 2 and s[1] < W:
            raise ProposalErr("wrong number of dimensions in initial ensemble distribution --- must be of dimension either (nParams,nWakers) or (nParams,2).")
        
        # output file?
        if outfile != None:
            f = open(outfile,'w')
        
        if s[1] == 2:
            for i in range(W):
                old_p.append((p0[:,1]-p0[:,0])*np.random.rand(npars)+p0[:,0])
                chain.append(old_p[-1])
                old_prob.append(logpost(old_p[-1],*args))
                if outfile != None:
                    for j in range(npars):
                        f.write('%.8e\t'%(old_p[-1][j]))
                    f.write('\n')
        else:
            for i in range(W):
                old_p.append(p0[:,i])
                chain.append(old_p[-1])
                old_prob.append(logpost(old_p[-1],*args))
                if outfile != None:
                    for j in range(npars):
                        f.write('%.8e\t'%(old_p[-1][j]))
                    f.write('\n')
        
        f.close()
        
        nacc = 0
        
        for it in range(N-1):
            for i in range(W):
                z = ((self.a-1.)*np.random.rand()+1)**2./self.a
                rint = np.random.randint(W-1)
                if rint >= i:
                    rint += 1
                new_p = old_p[rint]+z*(old_p[i]-old_p[rint])
                accept = False
                # if np.all(new_p > bounds[:,0]) and np.all(new_p < bounds[:,1]):
                new_prob = logpost(new_p,*args)
                diff = (npars-1.)*np.log(z)+new_prob-old_prob[i]
            
                if diff > 0:
                    accept = True
                else:
                    rn = np.random.rand()
                    if rn < np.exp(diff):
                        accept = True
                
                if accept:
                    old_prob[i] = new_prob
                    old_p[i] = new_p
                    nacc += 1
                
                if outfile != None:
                    f = open(outfile,'a')
                    for j in range(npars):
                        f.write('%10.8e\t'%(old_p[i][j]))
                    f.write('\n')
                    f.close()
                
                if it*W > burnin:
                    chain.append(old_p[i])
        
        acceptfrac = float(nacc)/N/W
        
        if acceptfrac > 0.1:
            print '''Ensemble sampling completed successfully:
    Acceptance fraction: %.3f
'''%(acceptfrac)
        else:
            print 'Warning: acceptance fraction < 10\%'
        
        return np.array(chain),acceptfrac
            
    