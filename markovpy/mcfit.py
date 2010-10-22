#!/usr/bin/env python
# encoding: utf-8
# 
#  markovpy.py
#  markovpy
#  
#  Created by Dan F-M on 2010-08-10.
# 

"""
mcfit...
"""

import sys

import numpy as np

from mcsampler import *

class MCError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def mcfit(loglike,bounds,args=None,sampler=None,proposal=None,N=10000,burnin=1000):
    """
    Fit args using MCMC and return samples from the PDF.
    """
    
    if sampler == None:
        sampler = MCSampler()
    
    try:
        bounds = np.array(bounds)
        
        if np.shape(bounds)[1] != 2:
            raise MCError("provide bounds on the parameter space")
        if proposal == None:
            proposal = (bounds[:,1]-bounds[:,0])/10.
        
        return sampler.sample_pdf(loglike,bounds,proposal,N,burnin,args)        
    except (MCError,ProposalErr) as e:
        print "MCMC exception raised with message: "+e.value    
