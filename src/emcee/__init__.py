# -*- coding: utf-8 -*-

__bibtex__ = """
@article{emcee,
   author = {{Foreman-Mackey}, D. and {Hogg}, D.~W. and {Lang}, D. and {Goodman}, J.},
    title = {emcee: The MCMC Hammer},
  journal = {PASP},
     year = 2013,
   volume = 125,
    pages = {306-312},
   eprint = {1202.3665},
      doi = {10.1086/670067}
}
"""
__uri__ = "https://emcee.readthedocs.io"
__author__ = "Daniel Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__license__ = "MIT"
__description__ = "The Python ensemble sampling toolkit for MCMC"


from .emcee_version import __version__  # isort:skip

from . import autocorr, backends, moves
from .ensemble import EnsembleSampler, walkers_independent
from .state import State

__all__ = [
    "EnsembleSampler",
    "walkers_independent",
    "State",
    "moves",
    "autocorr",
    "backends",
    "__version__",
]
