# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

__version__ = "3.0rc1"
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
"""  # NOQA

try:
    __EMCEE_SETUP__
except NameError:
    __EMCEE_SETUP__ = False

if not __EMCEE_SETUP__:
    from .ensemble import EnsembleSampler
    from .state import State

    from . import moves
    from . import autocorr
    from . import backends

    __all__ = ["EnsembleSampler", "State", "moves", "autocorr", "backends"]
