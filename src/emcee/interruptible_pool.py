# -*- coding: utf-8 -*-

# The standard library now has an interruptible pool
from multiprocessing.pool import Pool as InterruptiblePool

__all__ = ["InterruptiblePool"]
