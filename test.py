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

# define the true variance tensor
np.random.seed()
dim     = 20
vtensor = np.zeros([dim,dim])

# the variance tensor is the sum of outer-products of random vectors
for i in range(dim+1):
    v = np.random.rand(dim)/dim
    vtensor += np.outer(v,v)

# eigenvalues and eigenvectors of vtensor
# column evecs[:,i] is the eigenvector corresponding 
# to the eigenvalue eval[i].
evals,evecs = np.linalg.eig(vtensor)

# sqrt of 2 pi times determinant of vtensor
vtensor_det = np.sqrt(np.linalg.det(2.0*np.pi*vtensor))

# inverse of vtensor /2
vtensor_inv = -np.linalg.inv(vtensor)/2.0

def main():
    """Run a Markov chain Monte Carlo to fit a high-dimensional pdf"""
    
    # fit the data -- sigma sampled in log
    bounds = []
    for i in range(dim):
        bounds.append([-2.,2.])
        
    samps,frac = mcmc.mcfit(loglike,bounds,N=10**4)#,sampler=mcmc.mcsampler.MCSampler())
    vtens_est = np.cov(samps.T)
    print np.sum(vtens_est-vtensor)
    
    for i in range(dim):
        print evecs[:,i]
        x = np.dot(samps,evecs[:,i])
        x = x[x < 3.*np.sqrt(evals[i])]
        x = x[x > -3.*np.sqrt(evals[i])]
        pl.figure().add_subplot(111).hist(x,100,normed=True,histtype="step",color="k")
        
        xs = np.linspace(min(x),max(x),500)
        pl.plot(xs,np.exp(-xs**2/evals[i]/2)/np.sqrt(2*np.pi*evals[i]))
        
        pl.xlim([-3.*np.sqrt(evals[i]),3.*np.sqrt(evals[i])])
    
    pl.show()

def loglike(p):
    return np.dot(p,np.dot(vtensor_inv,p)) - np.log(vtensor_det)

if __name__ == '__main__':
    main()
