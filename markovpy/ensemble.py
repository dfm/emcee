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

# output file
# allowed file types:
#  ascii, hdf5

try:
    import h5py
    
    # names of hdf5 datasets
    MPHDF5Chain      = 'hdf5_chain'
    MPHDF5RState     = 'hdf5_rstate'
    MPHDF5NPars      = 'hdf5_npars'
    MPHDF5NWalkers   = 'hdf5_nwalkers'
    MPHDF5AParam     = 'hdf5_a'
    MPHDF5PostArgs   = 'hdf5_postargs'
    MPHDF5NAccept    = 'hdf5_naccept'
    MPHDF5Iterations = 'hdf5_iterations'
    
except:
    h5py = None

try:
    import multiprocessing
except:
    multiprocessing = None

class EnsembleSampler:
    """Ensemble sampling following Goodman & Weare (2009)"""
    def __init__(self,nwalkers,npars,lnposteriorfn,manylnposteriorfn=None,
                 postargs=(),a=2.,outfile=None,clobber=True,outtype='ascii',
                 threads=1):
        # multiprocessing
        self.threads = threads
        self.pool    = None
        if threads > 1 and multiprocessing is not None:
            self.pool = multiprocessing.Pool(threads)
        elif threads > 1:
            print "Warning: multiprocessing package isn't loaded"

        # Initialize a random number generator that we own
        self.random = np.random.mtrand.RandomState()
        
        # a function that returns the posterior pdf of interest
        assert(lnposteriorfn is None or manylnposteriorfn is None)
        assert(lnposteriorfn is not None or manylnposteriorfn is not None)
        self.lnposteriorfn = lnposteriorfn
        self.manylnposteriorfn = manylnposteriorfn
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
        self.outtype = outtype.lower()
        self.outfile = outfile
        self.clobber = clobber
        
        self.clear_chain()
    
    def clear_chain(self):
        """
        Empty/initialize the Markov chain place holders
        
        The shape of self.chain is [K,M,N] where
            - K is the number of walkers,
            - M is the number of parameters and
            - N is the number of steps taken
        """
        
        self.chain         = np.empty([self.nwalkers,self.npars,0],dtype=float)
        self.lnprobability = np.empty([self.nwalkers,0])
        self.iterations    = 0
        self.naccepted     = np.zeros(self.nwalkers)
        
        if self.outfile is not None and self.clobber:
            if os.path.exists(self.outfile):
                os.remove(self.outfile)
            if self.outtype == 'hdf5' and h5py is not None:
                f = h5py.File(self.outfile, 'w')
                f.create_dataset(MPHDF5Chain, [self.nwalkers,self.npars,1],
                    self.chain.dtype, maxshape=[self.nwalkers,self.npars,None])
                f.create_group(MPHDF5RState)
                for i,r0 in enumerate(self.random.get_state()):
                    f[MPHDF5RState]['%d'%i] = r0
                
                # this is how we'll read the random state back in... todo
                #
                # for r in f['hdf5_rstate']:
                #     print f['hdf5_rstate'][r][...]
                #
                
                f.create_group(MPHDF5PostArgs)
                if type(self.postargs) is np.ndarray:
                    f[MPHDF5PostArgs]['0'] = self.postargs
                else:
                    for i,r0 in enumerate(self.postargs):
                        f[MPHDF5PostArgs]['%d'%i] = r0
                
                f[MPHDF5NPars]    = self.npars
                f[MPHDF5NWalkers] = self.nwalkers
                f[MPHDF5AParam]   = self.a
                
                f[MPHDF5NAccept]    = self.naccepted
                f[MPHDF5Iterations] = self.iterations
                
                f.close()
    
    def ensemble_lnposterior(self, pos):
        """
        Returns a vector of ln-posterior values for each walker in the ensemble
        """
        if self.pool is not None:
            M = self.pool.map
        else:
            M = map
        return np.array(M(self.lnposteriorfn, [pos[i]
                    for i in range(self.nwalkers)]))
        
        #if self.manylnposteriorfn is not None:
        #    return self.manylnposteriorfn(pos, *self.postargs)
        #return np.array([self.lnposteriorfn(pos[i], *self.postargs)
        #                 for i in range(self.nwalkers)])

    
    def run_mcmc(self,position,randomstate,iterations,lnprobinit=None):
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
                or it can be None and the sampler will seed itself
            * "iterations" is the number of steps to perform
            * if you already know the ln-probability of your ensemble,
              you can provide that vector in "lnprobinit" otherwise,
              we'll calculate it
        
        Outputs:
            This function returns a tuple including the
            position vector in parameter space, the vector of
            ln-probabilities for each walker and the random number
            generator state current at the END OF THE RUN.
            The position and state values can then be fed right back
            into this function to take more steps.  To access the 
            values for the whole chain, use the accessor functions:
                self.get_lnprobability() and
                self.get_chain()
        
        """
        for pos,lnprob,state in self.sample(position,lnprobinit,randomstate,
                                          iterations=iterations):
            pass
        
        return pos,lnprob,state
    
    def sample(self,position0,lnprob,randomstate,*args,**kwargs):
        """We do the heavy lifting here"""
        
        # copy the original position so that it doesn't get over-written
        position = np.array(position0)
        
        # calculate the current probability
        if lnprob == None:
            lnprob = self.ensemble_lnposterior(position)
        
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
        
        # resize the chain array for speed (Thanks Hogg&Lang)
        assert(np.shape(self.chain)[-1] == self.iterations)
        self.chain = np.dstack((self.chain,
                            np.zeros([self.nwalkers,self.npars,iterations])))
        
        # resize the hdf5 file if needed
        if self.outtype == 'hdf5' and h5py is not None:
            f = h5py.File(self.outfile, 'a')
            f[MPHDF5Chain].resize((self.nwalkers,self.npars,
                                   self.iterations+iterations))
            f.close()
        
        # sample chain as an iterator
        for k in xrange(iterations):
            zz = ((self.a-1.)*self.random.rand(self.nwalkers)+1)**2./self.a
            
            rint = self.random.randint(self.nwalkers-1, size=(self.nwalkers,))
            # if you have to ask you won't understand the answer </evil>
            rint[rint >= np.arange(self.nwalkers)] += 1
            
            # propose new walker position and calculate the lnprobability
            newposition = position[rint] + \
                    zz[:,np.newaxis]*(position-position[rint])
            newposition[:,self.fixedinds] = self.fixedvals
            newlnprob = self.ensemble_lnposterior(newposition)
            lnpdiff = (self.neff - 1.) * np.log(zz) + newlnprob - lnprob
            accept = (lnpdiff > np.log(self.random.rand(self.nwalkers)))
            if any(accept):
                lnprob[accept] = newlnprob[accept]
                position[accept,:] = newposition[accept,:]
                self.naccepted[accept] += 1
            
            # append current position and lnprobability (of all walkers)
            # to the chain
            self.chain[:,:,self.iterations] = position
            self.lnprobability = np.concatenate((self.lnprobability.T,
                                               [lnprob]),axis=0).T
            
            # write the current position to disk
            if self.outfile is not None:
                self.write_step(position)
            
            self.iterations += 1
            yield position, lnprob, self.random.get_state()
    
    def write_step(self,position):
        """
        Write the current position vector to a file...
        
        """
        if self.outtype == 'ascii':
            f = open(self.outfile,'a')
            for k in range(self.nwalkers):
                for i in range(self.npars):
                    f.write('%10.8e\t'%(position[k,i]))
                f.write('\n')
            f.close()
        elif self.outtype == 'hdf5':
            assert(h5py is not None)
            f = h5py.File(self.outfile, 'a')
            f[MPHDF5Chain][:,:,self.iterations] = position
            f[MPHDF5NAccept][...] = self.naccepted
            f[MPHDF5Iterations][...] = self.iterations
            f.close()
    
    def save_state(self,fn=None):
        pass
    
    def acceptance_fraction(self):
        return self.naccepted/self.iterations
    
    def get_lnprobability(self):
        return self.lnprobability
    
    def get_chain(self):
        return self.chain
    
    def fix_parameters(self, inds, vals):
        assert (len(inds) == len(vals)), "len(inds) must equal len(vals)"
        
        self.fixedinds = np.array(inds)
        self.fixedvals = np.array(vals)
        self.neff = self.npars - len(inds)
    
    def clustering(self,position,lnprob,randomstate):
        """Clustering algorithm (REFERENCE) to avoid getting trapped"""
        # sort the walkers based on lnprobability
        if lnprob == None:
            lnprob = np.array([self.lnposteriorfn(position[i],self.postargs)
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
                lnprob[k] = self.lnposteriorfn(position[k],self.postargs)
        
        return position, lnprob, self.random.get_state()


