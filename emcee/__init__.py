# -*- coding: utf-8 -*-

__version__ = "3.0.0-dev"

try:
    __EMCEE_SETUP__
except NameError:
    __EMCEE_SETUP__ = False

if not __EMCEE_SETUP__:
    __all__ = ["proposals", "Sampler", "Ensemble",
               "BaseWalker", "SimpleWalker"]

    from . import proposals
    from .sampler import Sampler
    from .ensemble import Ensemble
    from .walker import BaseWalker, SimpleWalker
