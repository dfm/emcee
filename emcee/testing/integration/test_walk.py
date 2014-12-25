# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_normal_walk", "test_uniform_walk"]

from ... import proposals
from .test_proposal import _test_normal, _test_uniform


def test_normal_walk(**kwargs):
    _test_normal(proposals.WalkProposal(s=3), **kwargs)


def test_uniform_walk(**kwargs):
    _test_uniform(proposals.WalkProposal(s=3), **kwargs)
