#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

from functools import wraps


class memoized(object):

    def __init__(self, attr):
        self.attr = attr

    def __call__(self, f):
        @wraps(f)
        def wrapper(obj, *args, **kwargs):
            value = getattr(obj, self.attr)
            if value is None:
                value = f(obj, *args, **kwargs)
                setattr(obj, self.attr, value)
            return value
        return wrapper
