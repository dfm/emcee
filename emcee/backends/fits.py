# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["FITSBackend", "TempFITSBackend"]

import os
from tempfile import NamedTemporaryFile

import numpy as np

try:
    import fitsio
except ImportError:
    fitsio = None

from .backend import Backend
from .. import __version__


class FITSBackend(Backend):

    HDU_MAP = (None, "accepted", "chain", "log_prob", "blobs")

    def __init__(self, filename):
        if fitsio is None:
            raise ImportError("you must install 'fitsio' to use the "
                              "FITSBackend")
        self.filename = filename

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
        return fitsio.FITS(self.filename, mode, clobber=clobber)

    def reset(self, nwalkers, ndim):
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
        if name not in self.HDU_MAP:
            raise ValueError("unrecognized value name")
        with self.open() as f:
            hdr = f[0].read_header()
            iteration = hdr["ITERAT"]
            if iteration <= 0:
                raise AttributeError("You must run the sampler with "
                                     "'store == True' before accessing the "
                                     "results")

            if name == "blobs" and not hdr["BLOBS"]:
                return None

            v = f[self.HDU_MAP.index(name)].read()

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
        None
        # with self.open() as f:
        #     elements = [
        #         v
        #         for k, v in sorted(f[self.name].attrs.items())
        #         if k.startswith("random_state_")
        #     ]
        # return elements if len(elements) else None

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
            has_blobs = hdr["BLOBS"]
            if has_blobs:
                dtype = np.dtype((blobs[0].dtype, blobs[0].shape))

            # Deal with things on the first update
            if iteration == 0:
                if 1 in f:
                    fs = f
                else:
                    fs = [None, f, f, f, f]
                fs[1].write(np.zeros(nwalkers, dtype=int))
                fs[2].write(np.empty((ngrow, nwalkers, ndim), dtype=float))
                fs[3].write(np.empty((ngrow, nwalkers), dtype=float))
                if has_blobs:
                    fs[4].write(np.empty(ngrow, nwalkers), dtype=dtype)

            # Otherwise append
            else:
                i = ngrow - (len(f[1]) - iteration)
                f[2].append(np.empty((i, nwalkers, ndim), dtype=float))
                f[3].append(np.empty((i, nwalkers), dtype=float))
                if has_blobs:
                    f[4].append(np.empty(i, nwalkers), dtype=dtype)

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
            f[2].write(coords[None, :, :], firstrow=iteration)
            f[3].write(log_prob[None, :], firstrow=iteration)
            if blobs is not None:
                f[4].write(blobs[None, :], firstrow=iteration)

            f[0].write_key("ITERAT", iteration + 1)


class TempFITSBackend(object):

    def __enter__(self):
        f = NamedTemporaryFile("w", delete=False)
        f.close()
        self.filename = f.name
        return FITSBackend(f.name)

    def __exit__(self, exception_type, exception_value, traceback):
        os.remove(self.filename)
