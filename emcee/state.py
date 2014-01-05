#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["MemoizedState"]


class MemoizedState(object):

    def __init__(self, coords):
        self.set_coords(coords)

    @property
    def factory(self):
        return self.__class__

    def get_coords(self):
        return self.coords

    def set_coords(self, coords):
        self._lnprob = None
        self.coords = coords

    def get_lnprob(self):
        if self._lnprob is None:
            self._lnprob = self.lnprob(self.coords)
        return self._lnprob

    def lnprob(self, coords):
        raise NotImplementedError("Must be implemented by subclasses.")
