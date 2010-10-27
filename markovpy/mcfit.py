#!/usr/bin/env python
# encoding: utf-8

"""
 markovpy.py
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

import sys

import numpy as np

from mcsampler import *
from ensemble import EnsembleSampler

class MCError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def mcfit(logpost,bounds,args=(),sampler=None,proposal=None,N=10000,burnin=1000,outfile=None):
    """
    Fit args using MCMC and return samples from the PDF.
    """
    
    if sampler == None:
        # sampler = MCSampler()
        sampler = EnsembleSampler(100)
    
    try:
        bounds = np.array(bounds)
        
        # if np.shape(bounds)[1] != 2:
        #     raise MCError("provide bounds on the parameter space")
        if proposal == None:
            proposal = (bounds[:,1]-bounds[:,0])/10.
        
        return sampler.sample_pdf(logpost,bounds,proposal,N,burnin,args,outfile=outfile)
    except (MCError,ProposalErr) as e:
        print "MCMC exception raised with message: "+e.value    
