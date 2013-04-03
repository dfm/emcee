#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from emcee.sampler import *
from emcee.mh import *
from emcee.ensemble import *
from emcee.ptsampler import *
from emcee import utils


__version__ = "1.2.0"


def test():
    from .tests import Tests

    test_names = [
                  ("Parallel Sampler",    "test_parallel"),
                  ("Ensemble Sampler",    "test_ensemble"),
                  ("Metropolis-Hastings", "test_mh"),
                  ("Parallel Tempering", "test_pt_sampler"),
                  ("Pickling", "test_pickle")
                 ]

    print("Starting tests...")

    failures = 0
    tests = Tests()
    for t in test_names:
        tests.setUp()
        try:
            getattr(tests, t[1])()
        except Exception as e:
            print("Test: {0} failed with error:".format(t[0]))
            print("\t{0}: {1}".format(e.__class__.__name__, e))
            failures += 1
        else:
            print("Test: {0} passed.".format(t[0]))

    print("{0} tests passed".format(len(test_names) - failures))
    print("{0} tests failed".format(failures))
