# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["FileBackend"]

import os

from .. import __version__
from .base import BackendBase


class FileBackend(BackendBase):
    """A backend that stores the chain in  a file.

    This is a base class for file-based backends, not meant to be used directly.

    Args:
        filename (str): The name of the HDF5 file where the chain will be
            saved.
        read_only (bool; optional): If ``True``, the backend will throw a
            ``RuntimeError`` if the file is opened with write access.

    """

    def __init__(self, filename, read_only=False):
        """Initialize self given a file name.

        If ``read_only`` is ``True``, will throw a ``RuntimeError``
        if the file is opened with write access.
        """
        self.filename = filename
        self.read_only = read_only

    @property
    def initialized(self):
        """Return True if the backend has been initialized."""
        if not os.path.exists(self.filename):
            return False
        return True
