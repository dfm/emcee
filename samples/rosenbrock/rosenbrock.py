#!/usr/bin/env python
# encoding: utf-8
"""
 rosenbrock.py
 
 Created by Dan F-M on 11/11/2010.


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

def lnposterior(p):
    return -(100*(p[1]-p[0]*p[0])**2+(1-p[0])**2)/20.0

x1 = np.linspace(-4.0,6.0,500)
x2 = np.linspace(-2.0,30.0,500)
z = []
for xx1 in x1:
    tmp = []
    for xx2 in x2:
        tmp.append(np.exp(lnposterior([xx1,xx2])))
    z.append(tmp)
z = np.array(z)
pl.contour(x1,x2,z.T,colors='k')
ax = pl.gca()

pl.savefig('analytic.svg')

pl.figure()

np.random.seed()

nwalkers = 100
pos0 = []
for k in range(nwalkers):
    pos0.append(10.*np.random.randn(2))
pos0=np.array(pos0)

state0 = np.random.get_state()

sampler = mp.EnsembleSampler(nwalkers,2,lnposterior)
sampler2 = mp.EnsembleSampler(nwalkers,2,lnposterior)

# for position,prob,state in sampler.sample(pos0,None,state0,iterations=2000):
#     pass

for position2,prob2,state2 in sampler.sample(pos0,None,state0,iterations=500):
    pass
for position2,prob2,state2 in sampler2.sample(position2,prob2,state2,iterations=5000):
    pass

print np.mean(sampler2.acceptance_fraction())
# print np.sum(np.fabs(sampler.chain-sampler2.chain))

# see https://github.com/dfm/Python-Codebase/wiki/Plotting
import dfm.plotting
dfm.plotting.contour(sampler2.chain[:,0,:].flatten(),sampler2.chain[:,1,:].flatten(),bins=200)
pl.xlim(ax.get_xlim())
pl.ylim(ax.get_ylim())

pl.savefig('rosenbrock_samples.svg')

pl.figure()
time = mp.diagnostics.autocorrelation(sampler2.chain)
for k in range(nwalkers):
    pl.plot(time[k,0,:])

pl.figure()
for k in range(nwalkers):
    pl.plot(time[k,1,:])


pl.show()

