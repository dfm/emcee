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

This package requires [NumPy](http://numpy.scipy.org/) and it has been
tested on Python 2.6.5.


##USAGE

See the [wiki](http://github.com/dfm/MarkovPy/wiki) for information tutorials, documentation and sample code.


##LICENSE

MarkovPy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.

MarkovPy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MarkovPy.  If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
