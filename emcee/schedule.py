#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Schedule"]


class Schedule(object):

    def __init__(self, proposals):
        self.proposals = proposals

    def sample(self, ensemble):
        while True:
            for prop in self.proposals:
                ensemble, acceptance = prop.update(ensemble)
                yield ensemble
