# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["InterruptiblePool"]

# The standard library now has an interruptible pool
from multiprocessing.pool import Pool as InterruptiblePool
