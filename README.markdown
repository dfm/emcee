#MarkovPy

##NOTE

This is still a young module under active development so please use 
at your own risk and let me know what bugs you encounter.


##AUTHOR

Daniel Foreman-Mackey - dfm265 at nyu dot edu

If you find this code useful in your research, please let me know and
consider citing this package. Thanks!


##INTRODUCTION

MarkovPy is an extensible, pure-Python implementation of Markov chain
Monte Carlo (MCMC) curve fitting. The calling syntax is designed to be
like scipy.optimize so that it can (almost) be a drop in replacement.

There are currently no plans to port the included ensemble sampler to
PyMC or any other platforms but such ports would (of course) be welcomed.


##INSTALLATION

Navigate to this directory and run

`% ./setup.py install`

on the command line to install a module called markovpy in the default
Python path.


##DEPENDENCIES

This package requires [[NumPy|http://numpy.scipy.org/]] and it has been
tested on Python 2.6.5.


##USAGE

The sample script "test.py" shows an example of a high dimensional
Gaussian PDF.

The main function call is:

`samples,frac = markovpy.mcfit(logpost,p0,args=None,sampler=None,
                              proposal=None,N=1000,seed=None,outfile=None)`

Inputs:

logpost     -   a function logpost(params, *args) that returns the relative
                log-posterior for parameters "params".

p0          -   a (M,2) array which provides an initial guess of the bounds
                of the parameter space. NOTE: these bounds are not enforced
                beyond this initial guess. If you would like a hard prior
                on your parameter space, you must include it in your logpost
                function.
                
                or a (M,K) array providing the initial state of the
                ensemble --- preferably sampled from your prior distribution

sampler     -   a markovpy.mcsampler object that performs the sampling.
                The default sampler provided in 
                markovpy.ensemble.EnsembleSampler(K)
                is based on Goodman & Weare (2009) with K walkers.

seed        -   a seed value for numpy.random.  If you provide nothing,
                the sampler assumes that you have seeded the random
                number generator yourself.
                
outfile     -   a filename into which the tab-delimited Markov chain
                results can be written.  This file will be overwritten
                if it already exists.

Outputs:

samples     -   (N,M) array of samples from the posterior PDF

frac        -   the acceptance fraction of the chain


##LICENSE

MarkovPy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.

MarkovPy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MarkovPy.  If not, see [[http://www.gnu.org/licenses/]].
