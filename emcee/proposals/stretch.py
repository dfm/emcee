#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["StretchProposal"]

import copy
import numpy as np

from .proposals import Proposal


class StretchProposal(Proposal):

    def __init__(self, a=2.0):
        self.a = float(a)
