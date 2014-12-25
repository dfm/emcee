# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_normal_gaussian", "test_uniform_gaussian"]

from ... import proposals
from .test_proposal import _test_normal, _test_uniform


def test_normal_gaussian(**kwargs):
    _test_normal(proposals.GaussianProposal(0.5), **kwargs)


def test_uniform_gaussian(**kwargs):
    _test_uniform(proposals.GaussianProposal(0.5), **kwargs)
