# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

try:
    from schwimmbad import MPIPool
except ImportError:

    class MPIPool(object):

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The MPIPool from emcee has been forked to "
                "https://github.com/adrn/schwimmbad, "
                "please install that package to continue using the MPIPool"
            )

__all__ = ["MPIPool"]
