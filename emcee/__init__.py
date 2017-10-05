# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

__version__ = "3.0.0.dev0"

from .ensemble import EnsembleSampler

from . import moves
from . import autocorr

__all__ = ["EnsembleSampler", "moves", "autocorr"]


def test():
    from inspect import getmembers, ismethod
    from .tests import Tests

    print("Starting tests...")
    failures = 0
    tests = Tests()
    for o in getmembers(tests):
        tests.setUp()
        if ismethod(o[1]) and o[0].startswith("test"):
            print("{0} ...".format(o[0]))
            try:
                o[1]()
            except Exception as e:
                print("Failed with:\n    {0.__class__.__name__}: {0}"
                      .format(e))
                failures += 1
            else:
                print("    Passed.")

    print("{0} tests failed".format(failures))
