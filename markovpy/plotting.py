#!/usr/bin/env python
# encoding: utf-8
"""
 plotting.py

 Created by Dan F-M on 2010-11-06.


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
import scipy.special as sp

import pylab as pl
from matplotlib.colors import LinearSegmentedColormap


def plot2d(x,y,ax=None,bins=50,range=None,color='k'):
    if ax == None:
        ax = pl.gca()
    
    cz,cx,cy = np.histogram2d(x,y,bins=bins,range=range,normed=True)
    cx,cy = (cx[1:]+cx[:-1])/2.,(cy[1:]+cy[:-1])/2.
    
    # calculate probability confidence intervals
    z = np.sort(cz.flatten())[::-1]
    v = []
    n,nmax = 1.,2.
    s = 0.0
    sz = np.sum(z)
    for z0 in z:
        sig = sp.erf(n/np.sqrt(2))
        if s < sig*sz and s+z0 >= sig*sz:
            v.append(z0)
            n += 1.
            if n > nmax:
                break
        s += z0
    
    cz = cz.T
    
    ax.contour(cx,cy,cz,v,colors=color,linewidths=(2.,1.))
    ax.contourf(cx,cy,cz,[v[1],0.],cmap=LinearSegmentedColormap.from_list('cmap',(color,color),N=2),alpha=0.5)
    ax.contourf(cx,cy,cz,[v[0],0.],cmap=LinearSegmentedColormap.from_list('cmap',(color,color),N=2))
    
    if range != None:
        ax.set_xlim(range[0])
        ax.set_ylim(range[1])
