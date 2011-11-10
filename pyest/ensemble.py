# encoding: utf-8
"""
This is a Markov chain Monte Carlo (MCMC) sampler based on:

Goodman & Weare, Ensemble Samplers With Affine Invariance
   Comm. App. Math. Comp. Sci., Vol. 5 (2010), No. 1, 65–80

"""

from __future__ import division

__all__ = ['EnsembleSampler']

import os
import pickle

import numpy as np

import acor

try:
    import h5py

    # names of hdf5 datasets
    MPHDF5Chain       = 'chain'
    MPHDF5LnProb      = 'lnprob'
    MPHDF5RState      = 'rstate'
    MPHDF5NPars       = 'npars'
    MPHDF5NWalkers    = 'nwalkers'
    MPHDF5AParam      = 'a'
    MPHDF5PostArgs    = 'postargs'
    MPHDF5NAccept     = 'naccept'
    MPHDF5Iterations  = 'iterations'

except:
    h5py = None

try:
    import multiprocessing
except:
    multiprocessing = None

# Here, you will find some Python MAGIC that wraps functions in such a way to
# try to make them pickleable. This is important if you want to use multiprocessing

class _function_wrapper(object):
    def __init__(self, f, postargs):
        self.f = f
        self.postargs = postargs
    def __call__(self, x):
        return self.f(x, *self.postargs)

def _wrap_function(func,args):
    # check whether func is pickleable
    pickle.dumps(func,-1)
    if args == ():
        return func
    fw = _function_wrapper(func, args)
    pickle.dumps(fw,-1)
    return fw

class EnsembleSampler(object):
    """
    Ensemble sampling following Goodman & Weare (2010) with optional parallelization.

    Parameters
    ----------
    nwalkers : int
        Number of Goodman & Weare "walkers"

    npars : int
        Number of dimensions in parameter space

    lnposteriorfn : function
        A function that takes a vector in the parameter space as input and
        returns the ln-posterior for that position. If you want to use
        multiprocessing, lnposteriorfn *must* be pickleable.

    postargs : tuple
        Tuple of arguments for lnposteriorfn i.e. ``f(x,*args)``.
        Must be a tuple!

    a : float, optional
        The sampler scale (see [1]_)

    outfile : str, optional
        Filename for output.

    outtype : str, optional
        Type of output to write (options "ASCII" or "HDF5")

    clobber : bool, optional
        Overwrite file if it already exists? Otherwise, append. (default: True)

    threads : int, optional
        Number of threads to run. If you wish to run with >1 thread, the
        multiprocessing module must be installed in your Python path.

    pool : multiprocessing.Pool, optional
        The Pool to use.  If you do not provide one, one will be created for you.

    References
    ----------
    .. [1] J. Goodman and J. Weare, "Ensemble Samplers with Affine Invariance",
       Comm. App. Math. Comp. Sci., Vol. 5 (2010), No. 1, 65–80.

    """
    def __init__(self,nwalkers,npars,lnposteriorfn,postargs=(),
                 a=2.,outfile=None,clobber=True,outtype='ascii',
                 threads=1,pool=None,nensemble=2):
        if postargs is None:
            postargs = ()
        self.postargs = postargs

        # multiprocessing
        self._pool    = None
        if (threads > 1 or pool is not None) and multiprocessing is not None:
            # check to see if lnposteriorfn is pickleable
            try:
                self._lnposteriorfn = _wrap_function(lnposteriorfn, postargs)
            except pickle.PicklingError:
                print "Warning: Can't pickle lnposteriorfn, we'll only use 1 thread"
                threads = 1
            else:
                if pool is not None:
                    self._pool = pool
                else:
                    self._pool = multiprocessing.Pool(threads)
        elif threads > 1:
            print "Warning: multiprocessing package isn't loaded"
        if self._pool is None:
            self._lnposteriorfn = lambda x: lnposteriorfn(x,*postargs)

        # Initialize a random number generator that we own
        self._random = np.random.mtrand.RandomState()

        # the ensemble sampler parameters
        assert nwalkers > npars, \
            "You need more walkers than dim = %d"%(npars)
        self.npars    = npars
        self.nwalkers = nwalkers
        self.a        = a

        self._nensemble = nensemble
        self._ensembles = []
        for i in range(self._nensemble-1):
            self._ensembles.append(np.arange(i*self.nwalkers/self._nensemble,
                        (i+1)*self.nwalkers/self._nensemble, dtype=int))
        self._ensembles.append(np.arange(
                (self._nensemble-1)*self.nwalkers/self._nensemble,self.nwalkers,
                dtype=int))

        # used to fix some parameters to specific values for debugging purposes
        self._neff    = npars
        self._fixedinds = []
        self._fixedvals = []

        # optional output file, wipe it if it's already there
        self._outtype = outtype.lower()
        self._outfile = outfile
        self._clobber = clobber

        self.clear_chain()

    def clear_chain(self):
        """
        Clear the chain and some other stats so that the class can be reused

        This can be especially useful after a burn-in phase, for example.

        """
        self._chain         = np.empty([self.nwalkers,self.npars,0],dtype=float)
        self._lnprobability = np.empty([self.nwalkers,0])
        self._iterations    = 0
        self._naccepted     = np.zeros(self.nwalkers)

        if self._outfile is not None and self._clobber:
            if os.path.exists(self._outfile):
                os.remove(self._outfile)
            if self._outtype == 'hdf5' and h5py is not None:
                f = h5py.File(self._outfile, 'w')
                f.create_dataset(MPHDF5Chain, [self.nwalkers,self.npars,1],
                    self._chain.dtype, maxshape=[self.nwalkers,self.npars,None])
                f.create_dataset(MPHDF5LnProb, [self.nwalkers,1],
                    self._lnprobability.dtype, maxshape=[self.nwalkers,None])
                f.create_group(MPHDF5RState)
                for i,r0 in enumerate(self._random.get_state()):
                    f[MPHDF5RState]['%d'%i] = r0

                # this is how we'll read the random state back in... todo
                #
                # for r in f['rstate']:
                #     print f['rstate'][r][...]
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

                f[MPHDF5NAccept]    = self._naccepted
                f[MPHDF5Iterations] = self._iterations

                f.close()

    def ensemble_lnposterior(self, pos):
        """
        Return the ln-posterior probability for all the walkers

        This step is run in parallel if the sampler was initialized with > 1 threads.

        Parameters
        ----------
        pos : list (nwalkers, npars)
            A list of the positions of the walkers in the parameter space

        Returns
        -------
        lnposterior : list (nwalkers)
            A list of the log posterior values for each of the walkers

        """
        if self._pool is not None:
            M = self._pool.map
        else:
            M = map
        return np.array(M(self._lnposteriorfn, [pos[i]
                    for i in range(len(pos))]))

    def run_mcmc(self,position,randomstate,iterations,lnprob=None,lnprobinit=None):
        """
        Run a given number of MCMC steps

        If you want to run diagnostics between steps or only advance the chain
        by one step, use the EnsembleSampler.sample function

        Parameters
        ----------
        position : list (nwalkers, npars)
            A list of the positions of the walkers in the parameter space

        randomstate : tuple
            Returned by the get_state() function of numpy.random.mtrand.RandomState

        iterations : int
            Number of steps to run in the Markov chain

        lnprob : list (nwalkers), optional
            The list of log posterior probabilities for the walkers at positions
            given by the position parameter. If lnprobinit is None, the initial
            values are calculated using ensemble_lnposterior(position).

        lnprobinit : list (nwalkers), deprecated
            Superseded by lnprob.

        Returns
        -------
        pos : list (nwalkers, npars)
            The list of the _final_ positions of the walkers.

        lnprob : list (nwalkers)
            A list of all of the log posterior probabilities for the walkers at
            the _final_ position.

        state : tuple
            The state of the random number generator at the end of the run.

        """
        for pos,lnprob,state in self.sample(position,lnprob,randomstate,
                                          iterations=iterations):
            pass

        return pos,lnprob,state

    def _propose_position(self,s0,comp,lnprob):
        """
        Propose a new position given a current position and the complementary set

        Parameters
        ----------
        s0 : numpy.ndarray
            The current positions of the set of walkers that will be advanced

        comp : numpy.ndarray
            The complementary set of walkers

        lnprob : numpy.ndarray
            The current ln-probability of the s0 set

        Returns
        -------
        newposition : numpy.ndarray
            The new positions of the walkers (same shape as s0)

        newlnprob : numpy.ndarray
            The ln-probability of the walkers at newposition

        accept : numpy.ndarray
            Array of bools indicating whether or not a walker's new position
            should be accepted

        """
        s = np.atleast_2d(s0)
        n0 = len(s)
        if comp is not None:
            c = np.atleast_2d(comp)
        else:
            c = s
        ncomp = len(c)

        zz = ((self.a-1.)*self._random.rand(n0)+1)**2./self.a
        if comp is None:
            rint = self._random.randint(ncomp-1, size=(n0,))
            rint[rint >= np.arange(n0)] += 1
        else:
            rint = self._random.randint(ncomp, size=(n0,))

        # propose new walker position and calculate the lnprobability
        newposition = c[rint] - \
                zz[:,np.newaxis]*(c[rint]-s)
        newposition[:,self._fixedinds] = self._fixedvals
        newlnprob = self.ensemble_lnposterior(newposition)

        lnpdiff = (self._neff - 1.) * np.log(zz) + newlnprob - lnprob
        accept = (lnpdiff > np.log(self._random.rand(len(lnpdiff))))

        return newposition, newlnprob, accept

    def sample(self,position0,lnprob,randomstate,*args,**kwargs):
        """
        Advances the chain N steps as an iterator

        By default, it will only advance the chain by one step but if you're
        going to run multiple steps, the EnsembleSampler.run_mcmc or iterator
        version of this function have less overhead.

        Parameters
        ----------
        position0 : list (nwalkers, npars)
            A list of the initial positions of the walkers in the parameter space

        lnprob : list (nwalkers)
            The list of log posterior probabilities for the walkers at positions
            given by the position parameter. If it is None, the initial
            values are calculated using ensemble_lnposterior(position).

        randomstate : tuple
            Returned by the get_state() function of numpy.random.mtrand.RandomState

        iterations : int, optional
            Number of steps to run in the Markov chain

        Returns
        -------
        pos : list (nwalkers, npars)
            The list of the _final_ positions of the walkers.

        lnprob : list (nwalkers)
            A list of all of the log posterior probabilities for the walkers at
            the _final_ position.

        state : tuple
            The state of the random number generator at the end of the run.

        """
        # copy the original position so that it doesn't get over-written
        position = np.array(position0)

        # calculate the current probability
        if lnprob == None:
            lnprob = self.ensemble_lnposterior(position)

        # set the current state of our random number generator
        try:
            self._random.set_state(randomstate)
        except:
            self._random.seed()

        # how many iterations?  default to 1
        iterations = kwargs.pop('iterations', 1)

        # resize the chain array for speed (Thanks Hogg&Lang)
        binaryrep = self._outtype == 'hdf5' and h5py is not None
        if binaryrep:
            f = h5py.File(self._outfile, 'a')
            f[MPHDF5Chain].resize((self.nwalkers,self.npars,
                                   self._iterations+iterations))
            f[MPHDF5LnProb].resize((self.nwalkers,self._iterations+iterations))
            f.close()
        else:
            self._chain = np.dstack((self._chain,
                            np.zeros([self.nwalkers,self.npars,iterations])))
            self._lnprobability = np.concatenate((self._lnprobability,
                            np.zeros([self.nwalkers,iterations])),axis=-1)

        # sample chain as an iterator
        for k in xrange(iterations):
            for g in range(self._nensemble):
                # propose new walker position and calculate the lnprobability
                ens_in   = self._ensembles[g]
                ens_comp = self._ensembles[(g+1)%self._nensemble]
                newposition, newlnprob, accept = \
                        self._propose_position(position[ens_in],
                                position[ens_comp],
                                lnprob[ens_in])
                fullaccept = np.zeros(self.nwalkers,dtype=bool)
                fullaccept[ens_in] = accept
                if any(accept):
                    lnprob[fullaccept] = newlnprob[accept]
                    position[fullaccept] = newposition[accept]
                    self._naccepted[fullaccept] += 1

            # append current position and lnprobability (of all walkers)
            # to the chain
            if binaryrep:
                f = h5py.File(self._outfile, 'a')
                f[MPHDF5Chain][:,:,self._iterations] = position
                f[MPHDF5LnProb][:,self._iterations] = lnprob
                f[MPHDF5NAccept][...] = self._naccepted
                f[MPHDF5Iterations][...] = self._iterations
                f.close()
            else:
                self._chain[:,:,self._iterations] = position
                self._lnprobability = np.concatenate((self._lnprobability.T,
                                               [lnprob]),axis=0).T

            # write the current position to disk
            if self._outtype == 'ascii' and self._outfile is not None:
                self._write_step(position)

            self._iterations += 1
            yield position, lnprob, self._random.get_state()

    def _write_step(self,position):
        if self._outtype == 'ascii':
            f = open(self._outfile,'a')
            for k in range(self.nwalkers):
                for i in range(self.npars):
                    f.write('%10.8e\t'%(position[k,i]))
                f.write('\n')
            f.close()

    @property
    def acceptance_fraction(self):
        """
        Get a list of acceptance fractions for each walker in the ensemble

        Returns
        -------
        acceptance_fractions : list (nwalkers)
            The list of acceptance fractions for the walkers

        """
        return self._naccepted/self._iterations

    @property
    def lnprobability(self):
        return self.get_lnprobability()

    def get_lnprobability(self):
        """
        Get the set of ln-probabilities after running a MCMC

        Returns
        -------
        lnprob : numpy.ndarray (nwalkers, niterations)
            The ln-probabilities of each walker at each step

        """
        if self._outtype == 'hdf5' and h5py is not None:
            f = h5py.File(self._outfile)
            ret = f[MPHDF5LnProb][...]
            f.close()
            return ret
        return self._lnprobability

    @property
    def chain(self):
        return self.get_chain()

    def get_chain(self):
        """
        Get the MCMC samples

        Returns
        -------
        chain : numpy.ndarray (nwalkers, npars, niterations)
            The set of samples in the MCMC chain

        """
        if self._outtype == 'hdf5' and h5py is not None:
            f = h5py.File(self._outfile)
            ret = f[MPHDF5Chain][...]
            f.close()
            return ret
        return self._chain

    def fix_parameters(self, inds, vals):
        """
        Fix a set of parameters to specific values

        Parameters
        ----------
        inds : list
            List of the indices of the parameters that will be fixed

        vals : list
            List of the values to fix the parameters given by inds

        Raises
        ------
        AssertionError : If len(inds) doesn't equal len(vals)

        """
        assert (len(inds) == len(vals)), "len(inds) must equal len(vals)"

        self._fixedinds = np.array(inds)
        self._fixedvals = np.array(vals)
        self._neff = self.npars - len(inds)

    @property
    def acor(self):
        chain = self.chain
        tau = np.zeros(chain.shape[1])
        for i in range(chain.shape[1]):
            tau[i] = acor.acor(chain[:,i,:].flatten())
        return tau

    def _clustering(self,position,lnprob,randomstate):
        """
        Clustering algorithm (REFERENCE) to avoid getting trapped

        """
        # sort the walkers based on lnprobability
        if lnprob == None:
            lnprob = np.array([self._lnposteriorfn(position[i],self.postargs)
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
            self._random.set_state(randomstate)
        except:
            pass

        mean = np.mean(position[goodwalkers,:],axis=0)
        std  = np.std(position[goodwalkers,:],axis=0)

        for k in badwalkers:
            while big_mean-lnprob[k] > lnprob[k]-small_mean:
                position[k,:] = mean+std*self._random.randn(self.npars)
                lnprob[k] = self._lnposteriorfn(position[k],self.postargs)

        return position, lnprob, self._random.get_state()

