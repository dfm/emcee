# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_walker_pickle", "test_ensemble_pickle", "test_moves_pickle"]

import pickle
import numpy as np
from multiprocessing import Pool

from ... import moves, pools, Ensemble, SimpleWalker

from ..common import NormalWalker


def f(x):
    return 0.0


def test_walker_pickle():
    # Check to make sure that the standard walker types pickle.
    walker = NormalWalker(1.0)
    pickle.dumps(walker)

    # And the "simple" form with function pointers.
    walker = SimpleWalker(f, f)
    pickle.dumps(walker)


def test_ensemble_pickle(seed=1234):
    np.random.seed(seed)

    # The standard ensemble should pickle.
    e = Ensemble(NormalWalker(1.), np.random.randn(10, 3))
    s = pickle.dumps(e, -1)
    pickle.loads(s)

    # It should also work with a pool. NOTE: the pool gets lost in this
    # process.
    e = Ensemble(NormalWalker(1.0), np.random.randn(10, 3), pool=Pool())
    s = pickle.dumps(e, -1)
    pickle.loads(s)

    # It should also work with a pool. NOTE: the pool gets lost in this
    # process.
    e = Ensemble(NormalWalker(1.0), np.random.randn(10, 3),
                 pool=pools.InterruptiblePool())
    s = pickle.dumps(e, -1)
    pickle.loads(s)


def test_moves_pickle():
    for m in [moves.StretchMove(), moves.GaussianMove(1.0),
              moves.MHMove(None), moves.DEMove(1e-2), moves.WalkMove()]:
        s = pickle.dumps(m, -1)
        pickle.loads(s)
