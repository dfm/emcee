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
    def __init__(self,nwalkers,npars,lnposteriorfn,postargs=(),a=2.,filehandle=None):
        # Initialize a random number generator that we own
        self.random         = np.random.mtrand.RandomState()
        
        # a function that returns the posterior pdf of interest
        self.lnposteriorfn  = lnposteriorfn
        self.postargs       = postargs
        
        # the ensemble sampler parameters
        assert nwalkers > npars, "You need more walkers than the dimension of the space (%d)."%(npars)
        self.npars          = npars
        self.nwalkers       = nwalkers
        self.a              = a
        
        # chain
        self.chain          = np.empty([nwalkers,npars,0],dtype=float)
        self.probability    = np.empty([nwalkers,0])
        self.position       = None
        self.iterations     = 0
        self.naccepted      = np.zeros(nwalkers)
        
        # optional output file
        self.filehandle     = filehandle
    
    def run_mcmc(self,position,randomstate,iterations):
        for pos,state in self.sample(position,randomstate,iterations=iterations):
            pass
        
        return pos,state
    
    def sample(self,position,randomstate,*args,**kwargs):
        # calculate the current probability
        lnprob = np.array([self.lnposteriorfn(position[i],*(self.postargs)) for i in range(self.nwalkers)])
        
        # set the current state of our random number generator
        try:
            self.random.set_state(randomstate)
        except:
            self.random.seed()
        
        # how many iterations?  default to 1
        try:
            iterations = kwargs['iterations']
        except:
            iterations = 1
        
        # sample chain as an iterator
        for k in range(iterations):
            for i in range(self.nwalkers):
                z = ((self.a-1.)*self.random.rand()+1)**2./self.a
                rint = self.random.randint(self.nwalkers-1)
                if rint >= i:
                    rint += 1
                
                # propose new walker position and calculate the probability
                new_pos = position[rint]+z*(position[i]-position[rint])
                new_prob = self.lnposteriorfn(new_pos,*(self.postargs))
                
                # acceptance probability
                diff = (self.npars-1.)*np.log(z)+new_prob-lnprob[i]
                
                # do we accept it?
                accept = False
                if diff > 0:
                    accept = True
                else:
                    rn = self.random.rand()
                    if rn < np.exp(diff):
                        accept = True
                
                if accept:
                    lnprob[i] = new_prob
                    position[i] = new_pos
                    self.naccepted[i] += 1
            
            self.chain = np.dstack((self.chain, position))
            self.probability = np.concatenate((self.probability.T, [lnprob]),axis=0).T
            self.iterations += 1
            yield position, self.random.get_state()
    
    def acceptance_fraction(self):
        return self.naccepted/self.iterations
    

    