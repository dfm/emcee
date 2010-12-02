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
MarkovPy --- Easy Markov chain Monte Carlo

Author: Daniel Foreman-Mackey

Description: This is an extensible, pure-Python implementation of Markov
chain Monte Carlo (MCMC) curve fitting. The calling syntax is designed to
be like scipy.optimize so that it can (almost) be a drop in replacement.

Usage: The most basic usage is by the `mcfit` function but for more complete
control, the user should instantiate an `EnsembleSampler` object and
interact directly with the sampler.

"""

from mcfit import *
from mcsampler import *
from ensemble import *
import diagnostics

def get_version():
    return "0.0.2"

