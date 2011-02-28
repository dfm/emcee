#!/usr/bin/env python
# encoding: utf-8
"""
 line.py
 
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
import markovpy as mp

# data from Hogg et al.: http://arxiv.org/abs/1008.4686
#                  x    y   s_y   s_x   rho
data = np.array([[201, 592,  61,    9, -0.84],
                 [244, 401,  25,    4,  0.31],
                 [ 47, 583,  38,   11,  0.64],
                 [287, 402,  15,    7, -0.27],
                 [203, 495,  21,    5, -0.33],
                 [ 58, 173,  15,    9,  0.67],
                 [210, 479,  27,    4, -0.02],
                 [202, 504,  14,    4, -0.05],
                 [198, 510,  30,   11, -0.84],
                 [158, 416,  16,    7, -0.69],
                 [165, 393,  14,    5,  0.30],
                 [201, 442,  25,    5, -0.46],
                 [157, 317,  52,    5, -0.03],
                 [131, 311,  16,    6,  0.50],
                 [166, 400,  34,    6,  0.73],
                 [160, 337,  31,    5, -0.52],
                 [186, 423,  42,    9,  0.90],
                 [125, 334,  26,    8,  0.40],
                 [218, 533,  16,    6, -0.78],
                 [146, 344,  22,    5, -0.56]], dtype=float)

# linear "model"
line   = lambda x,p: p[0]*x+p[1]

def lnpost(p,data):
    """
    Posterior PDF from http://arxiv.org/abs/1008.4686
    assuming Gaussian uncertainties, flat priors on the
    model parameters and outlier rejection.
    """
    pb,yb,vb = p[2],p[3],np.exp(p[4])
    
    # prior 0 <= P_b <= 1
    if pb < 0 or pb > 1:
        return -np.inf
    
    # Gaussian uncertainties in y
    var1 = 2 * data[:,2]**2
    like = (1-pb) * np.exp(-(data[:,1]-line(data[:,0],p))**2/var1) / np.sqrt(var1*np.pi)
    
    # prune outliers
    varb = 2*vb + var1
    like += pb * np.exp(-(data[:,1]-yb)**2/varb) / np.sqrt(np.pi*varb)
    
    return np.sum(np.log(like))

# start with a really bad guess
np.random.seed()
nwalkers = 100
p0 = (np.array([10,500,0.5,200,-6])[np.newaxis,:])*np.random.rand(nwalkers*5).reshape([nwalkers,-1])

# run a Markov chain on our data
samps,post,frac = mp.mcfit(lnpost,p0,args=([data]),N=2000,burnin=200)


#pl.plot(mp.autocorrelation(samps))

# plot the data
pl.figure()
pl.errorbar(data[:,0],data[:,1],yerr=data[:,2],fmt='.k')

# slice out the slope and intersect
m = samps[:,0,:].flatten()
b = samps[:,1,:].flatten()

# plot the linear fit
xs = np.linspace(min(data[:,0]),max(data[:,0]),2)
pl.plot(xs,line(xs,[np.mean(m),np.mean(b)]),'k',lw=2.0)
pl.xlabel(r'$x$',fontsize=16.)
pl.ylabel(r'$y$',fontsize=16.)

pl.savefig('data_fit.png')

# see https://github.com/dfm/Python-Codebase/wiki/Plotting
try:
    # plot slope and intersect contours
    pl.figure()
    
    import dfm.plotting
    dfm.plotting.contour(m,b)
    pl.xlabel(r'$m$',fontsize=16.)
    pl.ylabel(r'$b$',fontsize=16.)

    pl.savefig('m_b.png')

    # plot the marginalized posterior of the outlier fraction
    pl.figure()
    pl.hist(samps[:,2,:].flatten(),100,histtype='step',color='k',lw=2.,normed=True)
    pl.gca().set_yticklabels([])
    pl.xlabel(r'$P_b$',fontsize=16.)

    pl.savefig('pb.png')

    pl.show()
except:
    print 'Check out https://github.com/dfm/Python-Codebase/wiki/Plotting for more plotting options'
