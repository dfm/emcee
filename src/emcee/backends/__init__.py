# -*- coding: utf-8 -*-

from .backend import Backend
from .hdf import HDFBackend, TempHDFBackend

__all__ = ["Backend", "HDFBackend", "TempHDFBackend", "get_test_backends"]


def get_test_backends():
    backends = [Backend]

    try:
        import h5py  # NOQA
    except ImportError:
        pass
    else:
        backends.append(TempHDFBackend)

    return backends
