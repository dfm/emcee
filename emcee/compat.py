# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["izip", "imap"]

try:
    from itertools import izip, imap
except ImportError:
    izip = zip
    imap = map
