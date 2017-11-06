# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["FITSBackend", "TempFITSBackend"]

import os
import pickle
from tempfile import NamedTemporaryFile

import numpy as np

try:
    import fitsio
except ImportError:
    fitsio = None

from .backend import Backend
from .. import __version__


class FITSBackend(Backend):
    """A backend that stores the chain in an FITS file using fitsio

    .. note:: You must install `fitsio <https://github.com/esheldon/fitsio>`_
        to use this backend.

    Args:
        filename (str): The name of the FITS file where the chain will be
            saved.
        pickle_filename (str; optional): The name of the file where the pickled
            random state be saved. By default, this is ``filename + ".pkl"``.
        read_only (bool; optional): If ``True``, the backend will throw a
            ``RuntimeError`` if the file is opened with write access.

    """

    def __init__(self, filename, pickle_filename=None, read_only=False):
        if fitsio is None:
            raise ImportError("you must install 'fitsio' to use the "
                              "FITSBackend")
        self.filename = filename
        if pickle_filename is None:
            pickle_filename = filename + ".pkl"
        self.pickle_filename = pickle_filename
        self.read_only = read_only

    @property
    def initialized(self):
        if not os.path.exists(self.filename):
            return False
        try:
            with self.open() as f:
                hdr = f[0].read_header()
                return bool(hdr.get("INIT", False))
        except (OSError, IOError):
            return False

    def open(self, mode="r", clobber=False):
        if self.read_only:
            if mode != "r" or clobber:
                raise RuntimeError("The backend has been loaded in read-only "
                                   "mode. Set `read_only = False` to make "
                                   "changes.")
        return fitsio.FITS(self.filename, mode, clobber=clobber)

    def reset(self, nwalkers, ndim):
        """Clear the state of the chain and empty the backend

        Args:
            nwakers (int): The size of the ensemble
            ndim (int): The number of dimensions

        """
        with self.open("rw", clobber=True) as f:
            header = dict(
                version=__version__,
                init=1,
                nwalkers=nwalkers,
                ndim=ndim,
                blobs=0,
                iterat=0,
            )
            f.write(None, header=header)

    def has_blobs(self):
        with self.open() as f:
            hdr = f[0].read_header()
            return bool(hdr["BLOBS"])

    def get_value(self, name, flat=False, thin=1, discard=0):
        with self.open() as f:
            hdr = f[0].read_header()
            iteration = hdr["ITERAT"]
            if iteration <= 0:
                raise AttributeError("You must run the sampler with "
                                     "'store == True' before accessing the "
                                     "results")

            if name == "blobs" and not hdr["BLOBS"]:
                return None

            v = f[name].read()

        v = v[discard+thin-1:self.iteration:thin]
        if flat:
            s = list(v.shape[1:])
            s[0] = np.prod(v.shape[:2])
            return v.reshape(s)
        return v

    @property
    def shape(self):
        with self.open() as f:
            hdr = f[0].read_header()
            return hdr["NWALKERS"], hdr["NDIM"]

    @property
    def iteration(self):
        with self.open() as f:
            hdr = f[0].read_header()
            return hdr["ITERAT"]

    @property
    def accepted(self):
        with self.open() as f:
            return f[1].read()

    @property
    def random_state(self):
        if not os.path.exists(self.pickle_filename):
            return None
        with open(self.pickle_filename, "rb") as f:
            return pickle.load(f)

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs: The current list of blobs. This is used to compute the
                dtype for the blobs array.

        """
        self._check_blobs(blobs)

        with self.open("rw") as f:
            hdr = f[0].read_header()
            iteration = hdr["ITERAT"]
            nwalkers = hdr["NWALKERS"]
            ndim = hdr["NDIM"]
            has_blobs = blobs is not None
            if has_blobs:
                dtype = np.dtype((blobs[0].dtype, blobs[0].shape))
            f[0].write_key("BLOBS", has_blobs)

            # Deal with things on the first update
            if iteration == 0:
                if 1 in f:
                    fs = f
                else:
                    fs = [None, f, f, f, f]
                fs[1].write(np.zeros(nwalkers, dtype=int), extname="accept")
                fs[2].write(np.zeros((ngrow, nwalkers, ndim), dtype=float),
                            extname="chain")
                fs[3].write(np.zeros((ngrow, nwalkers), dtype=float),
                            extname="log_prob")
                if has_blobs:
                    fs[4].write(np.zeros((ngrow, nwalkers), dtype=dtype),
                                extname="blobs")

            # Otherwise append
            else:
                hdr2 = f[2].read_header()
                i = ngrow - (hdr2["NAXIS3"] - iteration)
                f["chain"].write(np.zeros((i, nwalkers, ndim), dtype=float),
                                 start=(iteration, 0, 0))
                f["log_prob"].write(np.zeros((i, nwalkers), dtype=float),
                                    start=(iteration, 0))
                if has_blobs:
                    f["blobs"].append(np.zeros((i, nwalkers), dtype=dtype),
                                      start=(iteration, 0))

    def save_step(self, coords, log_prob, blobs, accepted, random_state):
        """Save a step to the file

        Args:
            coords (ndarray): The coordinates of the walkers in the ensemble.
            log_prob (ndarray): The log probability for each walker.
            blobs (ndarray or None): The blobs for each walker or ``None`` if
                there are no blobs.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.
            random_state: The current state of the random number generator.

        """
        self._check(coords, log_prob, blobs, accepted)

        with self.open("rw") as f:
            hdr = f[0].read_header()
            iteration = hdr["ITERAT"]

            f[1].write(f[1].read() + accepted)
            f[2].write(coords[None, :, :], start=(iteration, 0, 0))
            f[3].write(log_prob[None, :], start=(iteration, 0))
            if blobs is not None:
                start = [iteration] + [0] * len(blobs.shape)
                f[4].write(blobs[None, :], start=start)

            f[0].write_key("ITERAT", iteration + 1)

        with open(self.pickle_filename, "wb") as f:
            pickle.dump(random_state, f, -1)


class TempFITSBackend(object):

    def __enter__(self):
        f1 = NamedTemporaryFile("w", delete=False)
        f1.close()
        f2 = NamedTemporaryFile("w", delete=False)
        f2.close()
        self.filename = f1.name
        self.pickle_filename = f2.name
        return FITSBackend(f1.name, f2.name)

    def __exit__(self, exception_type, exception_value, traceback):
        os.remove(self.filename)
        os.remove(self.pickle_filename)
