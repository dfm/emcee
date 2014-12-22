# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["DefaultPool"]

try:
    from itertools import imap
except ImportError:
    imap = map


class DefaultPool(object):

    def map(self, fn, iterable):
        return imap(fn, iterable)
