# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "Backend",
    "HDFBackend", "TempHDFBackend",
    "get_test_backends",
]

from .backend import Backend
from .hdf import HDFBackend, TempHDFBackend


def get_test_backends():
    backends = [Backend]

    try:
        import h5py  # NOQA
    except ImportError:
        pass
    else:
        backends.append(TempHDFBackend)

    return backends
