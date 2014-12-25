# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_normal_stretch", "test_uniform_stretch"]

from ... import moves
from .test_proposal import _test_normal, _test_uniform


def test_normal_stretch(**kwargs):
    _test_normal(moves.StretchMove(), **kwargs)


def test_uniform_stretch(**kwargs):
    _test_uniform(moves.StretchMove(), **kwargs)
