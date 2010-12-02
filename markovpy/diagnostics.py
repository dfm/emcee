#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2010 Daniel Foreman-Mackey
# 
# This is part of MarkovPy.
# 
# MarkovPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
# 
# MarkovPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with MarkovPy.  If not, see <http://www.gnu.org/licenses/>.
#

"""
 diagnostics.py

 Created by Dan F-M on 2010-10-18.

"""

#from _C_diagnostics import *


import numpy as np


def autocorrelation(chain):
    means = np.mean(chain,axis=-1)
    variance = np.var(chain,axis=-1)
    
    shape = np.shape(chain)
    time = np.empty((shape[0],shape[1],0))
    N = shape[-1]
    
    for k in range(N/2):
        # this is only going to work for the ensemble sampler... fix this!
        time = np.dstack((time,np.sum((chain[:,:,:N-k]-means[:,:,np.newaxis])*(chain[:,:,k:]-means[:,:,np.newaxis]),axis=-1)[:,:,np.newaxis]))
    
    return np.array(time)/variance[:,:,np.newaxis]

