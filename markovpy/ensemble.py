#!/usr/bin/env python
# encoding: utf-8

"""
 ensemble.py

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

import os
import numpy as np

from mcsampler import MCSampler

class EnsembleSampler(MCSampler):
    """Ensemble sampling following Goodman & Weare (2009)"""
    def __init__(self,nwalkers,npars,lnposteriorfn,postargs=(),a=2.,
                 outfile=None,clobber=True):
        # Initialize a random number generator that we own
        self.random = np.random.mtrand.RandomState()
        
        # a function that returns the posterior pdf of interest
        self.lnposteriorfn = lnposteriorfn
        self.postargs      = postargs
        
        # the ensemble sampler parameters
        assert nwalkers > npars, \
            "You need more walkers than dim = %d"%(npars)
        self.npars    = npars
        self.nwalkers = nwalkers
        self.a        = a
        self.neff     = npars
        
        self.fixedinds = []
        self.fixedvals = []
        
        # optional output file, wipe it if it's already there
        self.outfile = outfile
        if outfile != None and clobber:
            if os.path.exists(outfile):
                os.remove(outfile)
        
        self.clear_chain()
    
    def clear_chain(self):
        """
        Empty/initialize the Markov chain place holders
        
        The shape of self.chain is [K,M,N] where
            - K is the number of walkers,
            - M is the number of parameters and
            - N is the number of steps taken
        """
        
        self.chain       = np.empty([self.nwalkers,self.npars,0],dtype=float)
        self.probability = np.empty([self.nwalkers,0])
        self.iterations  = 0
        self.naccepted   = np.zeros(self.nwalkers)
        
    
    def run_mcmc(self,position,randomstate,iterations):
        """
        Run a given number of MCMC steps
        
        Inputs:
            * "position" is an array with shape [K,M] where
                - K is the number of walkers and
                - M is the number of parameters
            * "randomstate" is the state of an instance of a
                NumPy random number generator.  You can access
                it with:
            
                    numpy.random.mtrand.RandomState().get_state()
            * "iterations" is the number of steps to perform
        
        Outputs:
            This function returns a tuple including the current
            position vector in parameter space, the vector of
            probabilities for each walker and the random number
            generator state.  The position and state values can
            then be fed right back into this function to take
            more steps.
        
        """
        for pos,prob,state in self.sample(position,None,randomstate,
                                          iterations=iterations):
            pass
        
        return pos,prob,state
    
    def sample(self,position0,lnprob,randomstate,*args,**kwargs):
        """We do the heavy lifting here"""
        
        # copy the original position so that it doesn't get over-written
        position = np.array(position0)
        
        # calculate the current probability
        if lnprob == None:
            lnprob = np.array([float(self.lnposteriorfn(position[i],
                                                  *(self.postargs)))
                               for i in range(self.nwalkers)])
        
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
        for k in xrange(iterations):
            for i in xrange(self.nwalkers):
                z = ((self.a-1.)*self.random.rand()+1)**2./self.a
                rint = self.random.randint(self.nwalkers-1)
                if rint >= i:
                    rint += 1
                
                # propose new walker position and calculate the probability
                new_pos = position[rint]+z*(position[i]-position[rint])
                new_pos[self.fixedinds] = self.fixedvals
                new_prob = self.lnposteriorfn(new_pos,*(self.postargs))
                
                accepted = False
                if new_prob > -np.inf:
                    # acceptance probability
                    diff = (self.neff-1.)*np.log(z)+new_prob-lnprob[i]
                    
                    # do we accept it?
                    if diff > 0:
                        accepted = True
                    else:
                        rn = self.random.rand()
                        if rn < np.exp(diff):
                            accepted = True
                
                if accepted:
                    # update chain if this step was accepted
                    lnprob[i] = new_prob
                    position[i] = new_pos
                    self.naccepted[i] += 1
            
            # append current position and probability (of all walkers)
            # to the chain
            self.chain = np.dstack((self.chain, position))
            self.probability = np.concatenate((self.probability.T,
                                               [lnprob]),axis=0).T
            
            # write the current position to disk
            self.write_step(position)
            self.iterations += 1
            yield position, lnprob, self.random.get_state()
    
    def write_step(self,position):
        """
        Write the current position vector to an ASCII file...
        
        dumb dumb dumb...
        
        """
        if self.outfile != None:
            f = open(self.outfile,'a')
            for k in range(self.nwalkers):
                for i in range(self.npars):
                    f.write('%10.8e\t'%(position[k,i]))
                f.write('\n')
            f.close()
    
    def acceptance_fraction(self):
        return self.naccepted/self.iterations
    
    def fix_parameters(self, inds, vals):
        assert (len(inds) == len(vals)), "len(inds) must equal len(vals)"
        
        self.fixedinds = np.array(inds)
        self.fixedvals = np.array(vals)
        self.neff = self.npars - len(inds)
    
    def clustering(self,position,lnprob,randomstate):
        """Clustering algorithm (REFERENCE) to avoid getting trapped"""
        # sort the walkers based on probability
        if lnprob == None:
            lnprob = np.array([self.lnposteriorfn(position[i],
                                                  *(self.postargs))
                               for i in range(self.nwalkers)])
        inds = np.argsort(lnprob)[::-1]
        
        for i,ind in enumerate(inds):
            if i > 0 and i < len(lnprob)-1:
                big_mean   = np.mean(lnprob[inds[:i]])
                small_mean = np.mean(lnprob[inds[i+1:]])
                if big_mean-lnprob[ind] > lnprob[ind]-small_mean:
                    break
        
        # which walkers are in the right place
        goodwalkers = inds[:i]
        badwalkers  = inds[i:]
        
        if len(badwalkers) > 1:
            print "Clustering: %d walkers rejected"%(len(badwalkers))
        elif len(badwalkers) == 1:
            print "Clustering: 1 walker rejected"
        
        # reasample the positions of the bad walkers
        # assuming that the right ones form a Gaussian
        try:
            self.random.set_state(randomstate)
        except:
            pass
        
        mean = np.mean(position[goodwalkers,:],axis=0)
        std  = np.std(position[goodwalkers,:],axis=0)
        
        for k in badwalkers:
            while big_mean-lnprob[k] > lnprob[k]-small_mean:
                position[k,:] = mean+std*self.random.randn(self.npars)
                lnprob[k] = self.lnposteriorfn(position[k],*(self.postargs))
        
        return position, lnprob, self.random.get_state()


