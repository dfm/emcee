#!/usr/bin/env python
# encoding: utf-8

"""
 mcsampler.py
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

class ProposalErr(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class MCSampler:
    """Metropolis-Hastings sampler using simulated annealing"""
    def __init__(self):
        pass
    
    def sample_pdf(self,logpost,bounds,proposal,N,burnin,args,seed=None,output=None):
        if seed != None:
            np.random.seed(seed)
        
        npars = np.shape(bounds)[0]
        jr = np.zeros([npars,npars])
        
        if np.shape(proposal)[0] == npars:
            if len(np.shape(proposal)) == 1:
                jr = np.diag(proposal)
            elif np.shape(proposal)[1] == npars:
                jr = proposal
            else:
                raise ProposalErr("wrong number of dimensions in proposal distribution")
        
        # run chain with annealing
        warmup = True
        temp = 1.0
        
        old_p = (bounds[:,1]-bounds[:,0])*np.random.rand(npars)+bounds[:,0]
        old_prob = logpost(old_p,*args)
        
        chain = []
        
        while temp >= 1.0:
            chain = [old_p]
            nacc = 0
            sigma = 0.
            sigma2 = 0.
            
            for it in range(N):
                new_p = old_p+np.dot(jr,np.random.randn(npars))
                accept = False
                # if np.all(new_p > bounds[:,0]) and np.all(new_p < bounds[:,1]):
                new_prob = logpost(new_p,*args)
                if np.exp(new_prob) > 0.:
                    sigma += new_prob
                    sigma2 += new_prob**2
                    # print it, new_prob, sigma/it, sigma2/it,np.sqrt(sigma2/it-sigma**2./it**2.)
                
                diff = (new_prob-old_prob)/temp
                
                if diff > 0:
                    accept = True
                else:
                    rn = np.random.rand()
                    if rn < np.exp(diff):
                        accept = True
                
                if accept:
                    old_prob = new_prob
                    old_p = new_p
                    nacc += 1
                
                if it > burnin:
                    chain.append(old_p)
                
            if warmup:
                warmup = False
                maxt = 2.**np.floor(np.log2(np.sqrt(sigma2/N-(sigma/N)**2.)))
                temp = maxt
            else:
                temp /= 2.
                covar = np.cov((np.array(chain)).T)
                chain = np.array(chain)
                # print 'covar = ',covar #/4.*2.4**2/npars
                
                try:
                    jr = 2.4*np.linalg.cholesky(covar/4./npars)
                    # print 'jr = ',jr
                except:# LinAlgError:
                    print 'Cholesky decomposition failed.'
        
        acceptfrac = float(nacc)/N
        
        if acceptfrac > 0.1:
            print '''mcfit completed successfully:
    Acceptance fraction: %.3f
    Maximum temperature: %.0f
'''%(acceptfrac,maxt)
        else:
            print 'Warning: acceptance fraction < 10\%'
        
        return np.array(chain),np.array([]),acceptfrac

    